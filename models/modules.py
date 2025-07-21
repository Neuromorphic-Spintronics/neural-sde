"""
Neural network modules for the neural SDE framework.

This module provides reusable neural network components that integrate with the parameter system and follow the codebase's architectural patterns. All modules use the established device management and parameter validation.
"""

from __future__ import annotations

from typing import Callable
import torch
from torch import nn, Tensor

from config import DEVICE
from parameters.hyperparameters import NetworkArchitecture
from .protocols import NetworkProtocol

class FeedForwardNetwork(nn.Sequential, NetworkProtocol):
    """
    Fully connected neural network with configurable activation functions and Xavier initialisation.
    
    This class provides a standard feedforward architecture that integrates with the parameter system's NetworkArchitecture specification.
    
    The network applies Xavier initialisation in good approximation to all layers and includes configurable activation functions between layers (except the final layer).
    
    Args:
        architecture: Network architecture specification defining layer dimensions
        activation: Activation function. If None, defaults to nn.Tanh.
                    Should return a new activation instance when called.
        device: Computation device. Defaults to globally configured device.
        
    Example:
        >>> from parameters.hyperparameters import NetworkArchitecture
        >>> architecture = NetworkArchitecture(input_size=3, hidden_sizes=[64, 32], output_size=2)
        >>> network = FeedForwardNetwork(architecture, activation=lambda: nn.ReLU())
        >>> output = network(torch.randn(10, 3))  # shape: [10, 2]

    """

    def __init__(
        self, 
        architecture: NetworkArchitecture, 
        activation: Callable[[], nn.Module] = nn.Tanh,
        device: torch.device = DEVICE,
    ) -> None:
        """
        Initialise the feedforward network with specified architecture.
        
        Args:
            architecture: Network layer configuration
            activation: Activation function
            device: Device for computation
        """
        
        self.architecture = architecture
        self.layer_sizes = architecture.layer_sizes
        
        layers = self._build_layers(activation)
        super().__init__(*layers)
        
        self._initialise_weights()
        self.to(device)

    def _build_layers(self, activation: Callable[[], nn.Module]) -> list[nn.Module]:
        """
        Construct the sequential layers for the network.
        
        Args:
            activation: Activation function
            
        Returns:
            List of layers to be passed to nn.Sequential
        """
        raise NotImplementedError()  # type: ignore

    def _initialise_weights(self) -> None:
        """
        Apply Xavier initialisation to all linear layers.
        
        This initialisation helps maintain gradient magnitudes through the network.
        """
        raise NotImplementedError()  # type: ignore

    def forward(self, input: Tensor) -> Tensor:  # type: ignore[override]
        """
        Forward pass through the network.
        
        Args:
            input: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
            
        Raises:
            ValueError: If input dimension doesn't match architecture (maybe this will be raised by PyTorch in any case)
        """
        raise NotImplementedError()  # type: ignore


def count_network_parameters(network: nn.Module) -> int:
    """
    Count the total number of trainable parameters in a network.
    
    This utility function provides a consistent way to count parameters across different network architectures for logging and analysis.
    
    Args:
        network: PyTorch neural network module
        
    Returns:
        Total number of trainable parameters
        
    Example:
        >>> network = FeedForwardNetwork(architecture)
        >>> param_count = count_network_parameters(network)
        >>> print(f"Network has {param_count:,} trainable parameters")
    """
    raise NotImplementedError()  # type: ignore


def initialise_network_weights(
    network: nn.Module, 
    initialisation_method: str = "xavier"
) -> None:
    """
    Apply weight initialisation to all linear layers in a network.
    
    Args:
        network: Neural network to initialise
        initialisation_method: Method to use ("xavier", "kaiming")
        
    Raises:
        ValueError: If initialisation method is not recognised
    """
    raise NotImplementedError()  # type: ignore
