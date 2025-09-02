"""Test configuration and fixtures for Torch-Playground."""

from typing import Tuple

import numpy as np
import pytest
import torch


@pytest.fixture
def random_seed():
    """Fix random seed for reproducible tests."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def sample_cifar_batch() -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a sample CIFAR-10 batch for testing."""
    batch_size = 4
    images = torch.randn(batch_size, 3, 32, 32)
    labels = torch.randint(0, 10, (batch_size,))
    return images, labels


@pytest.fixture
def sample_mnist_batch() -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a sample MNIST batch for testing."""
    batch_size = 4
    images = torch.randn(batch_size, 1, 28, 28)
    labels = torch.randint(0, 10, (batch_size,))
    return images, labels


@pytest.fixture
def device():
    """Get available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
