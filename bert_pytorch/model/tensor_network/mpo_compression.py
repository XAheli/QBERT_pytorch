import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import math

class MPODecomposition:
    """
    Matrix Product Operator decomposition using SVD-based energy retention.
    Based on CompactifAI methodology for tensor network compression.
    """
    
    def __init__(self, energy_threshold: float = 0.9):
        """
        Initialize MPO decomposition with energy retention threshold.
        
        Args:
            energy_threshold: Fraction of energy to retain (0.9 = 90%)
        """
        self.energy_threshold = energy_threshold
    
    def decompose(self, weight_matrix: torch.Tensor, max_bond_dim: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Decompose weight matrix using SVD with energy-based truncation.
        
        Args:
            weight_matrix: Input weight matrix [m, n]
            max_bond_dim: Maximum bond dimension (optional constraint)
            
        Returns:
            A: Left factor matrix
            B: Right factor matrix  
            stats: Decomposition statistics
        """
        # Perform SVD decomposition
        U, S, Vt = torch.svd(weight_matrix)
        
        # Calculate cumulative energy
        energy = S.pow(2)
        cumulative_energy = torch.cumsum(energy, dim=0)
        total_energy = cumulative_energy[-1]
        energy_ratios = cumulative_energy / total_energy
        
        # Find truncation point based on energy threshold
        truncation_idx = torch.where(energy_ratios >= self.energy_threshold)[0]
        if len(truncation_idx) > 0:
            bond_dim = min(truncation_idx[0].item() + 1, len(S))
        else:
            bond_dim = len(S)
            
        # Apply max bond dimension constraint if specified
        if max_bond_dim is not None:
            bond_dim = min(bond_dim, max_bond_dim)
        
        # Truncate factors
        U_trunc = U[:, :bond_dim]
        S_trunc = S[:bond_dim]
        Vt_trunc = Vt[:bond_dim, :]
        
        # Create factor matrices A and B
        A = U_trunc * torch.sqrt(S_trunc).unsqueeze(0)
        B = torch.sqrt(S_trunc).unsqueeze(1) * Vt_trunc
        
        # Calculate reconstruction error
        reconstructed = torch.mm(A, B)
        reconstruction_error = torch.norm(weight_matrix - reconstructed) / torch.norm(weight_matrix)
        
        # Calculate compression statistics
        original_params = weight_matrix.numel()
        compressed_params = A.numel() + B.numel()
        compression_ratio = compressed_params / original_params
        
        stats = {
            'bond_dimension': bond_dim,
            'original_shape': weight_matrix.shape,
            'factor_A_shape': A.shape,
            'factor_B_shape': B.shape,
            'original_parameters': original_params,
            'compressed_parameters': compressed_params,
            'compression_ratio': compression_ratio,
            'parameter_reduction': 1 - compression_ratio,
            'reconstruction_error': reconstruction_error.item(),
            'energy_retained': energy_ratios[bond_dim-1].item() if bond_dim > 0 else 0.0
        }
        
        return A, B, stats

class MPOLinear(nn.Module):
    """
    Linear layer with MPO decomposition for weight compression.
    Drop-in replacement for nn.Linear with tensor network compression.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 bond_dimension: Optional[int] = None, energy_threshold: float = 0.9):
        """
        Initialize MPOLinear layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension  
            bias: Whether to include bias term
            bond_dimension: Fixed bond dimension (optional)
            energy_threshold: Energy retention threshold for SVD
        """
        super(MPOLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.bond_dimension = bond_dimension
        self.energy_threshold = energy_threshold
        
        # Initialize factor matrices with Xavier uniform
        if bond_dimension is None:
            # Start with small default bond dimension
            bond_dimension = min(64, min(in_features, out_features) // 2)
            
        self.factor_A = nn.Parameter(torch.empty(out_features, bond_dimension))
        self.factor_B = nn.Parameter(torch.empty(bond_dimension, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform."""
        nn.init.xavier_uniform_(self.factor_A)
        nn.init.xavier_uniform_(self.factor_B)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    @classmethod
    def from_linear(cls, linear_layer: nn.Linear, bond_dimension: Optional[int] = None, 
                   energy_threshold: float = 0.9) -> 'MPOLinear':
        """
        Create MPOLinear from existing Linear layer using SVD decomposition.
        
        Args:
            linear_layer: Existing nn.Linear layer
            bond_dimension: Target bond dimension  
            energy_threshold: Energy retention threshold
            
        Returns:
            MPOLinear layer with decomposed weights
        """
        # Create new MPO layer
        mpo_layer = cls(
            linear_layer.in_features, 
            linear_layer.out_features,
            bias=linear_layer.bias is not None,
            bond_dimension=bond_dimension,
            energy_threshold=energy_threshold
        )
        
        # Decompose original weight matrix
        decomposer = MPODecomposition(energy_threshold)
        A, B, stats = decomposer.decompose(linear_layer.weight.data, bond_dimension)
        
        # Set decomposed weights
        mpo_layer.factor_A.data = A
        mpo_layer.factor_B.data = B
        
        # Copy bias if present
        if linear_layer.bias is not None:
            mpo_layer.bias.data = linear_layer.bias.data.clone()
            
        # Store decomposition statistics
        mpo_layer.compression_stats = stats
        
        return mpo_layer
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using factorized computation: (input @ B^T) @ A^T
        
        Args:
            input: Input tensor [..., in_features]
            
        Returns:
            Output tensor [..., out_features]
        """
        # Efficient factorized computation
        intermediate = torch.matmul(input, self.factor_B.t())
        output = torch.matmul(intermediate, self.factor_A.t())
        
        if self.bias is not None:
            output = output + self.bias
            
        return output
    
    def get_compression_ratio(self) -> float:
        """Calculate compression ratio compared to dense layer."""
        dense_params = self.in_features * self.out_features
        if self.bias is not None:
            dense_params += self.out_features
            
        compressed_params = self.factor_A.numel() + self.factor_B.numel()
        if self.bias is not None:
            compressed_params += self.bias.numel()
            
        return compressed_params / dense_params
    
    def extra_repr(self) -> str:
        """String representation with compression info."""
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bond_dimension={self.factor_A.shape[1]}, compression_ratio={self.get_compression_ratio():.4f}'

