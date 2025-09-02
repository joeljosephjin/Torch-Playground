"""Tests for data loading utilities."""

import pytest
import torch
from torch.utils.data import DataLoader

from data.data import load_cifar_10, load_cifar_10_other, load_mnist


class TestCIFAR10Loading:
    """Test CIFAR-10 data loading functions."""

    def test_load_cifar_10_basic(self):
        """Test basic CIFAR-10 loading functionality."""
        train_loader, test_loader = load_cifar_10(
            batch_size=4, perc_size=0.01
        )  # Use small subset

        # Test train loader
        assert isinstance(train_loader, DataLoader)
        train_batch = next(iter(train_loader))
        assert len(train_batch) == 2  # images, labels

        images, labels = train_batch
        assert images.shape[1:] == (3, 32, 32)  # CIFAR-10 dimensions
        assert images.shape[0] <= 4  # batch size
        assert labels.shape[0] == images.shape[0]
        assert torch.all(labels >= 0) and torch.all(labels < 10)  # Valid class indices

        # Test test loader
        assert isinstance(test_loader, DataLoader)
        test_batch = next(iter(test_loader))
        test_images, test_labels = test_batch
        assert test_images.shape[1:] == (3, 32, 32)

    def test_load_cifar_10_other_basic(self):
        """Test alternative CIFAR-10 loading functionality."""
        train_loader, test_loader = load_cifar_10_other(batch_size=4)

        # Test basic functionality
        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

        train_batch = next(iter(train_loader))
        images, labels = train_batch
        assert images.shape[1:] == (3, 32, 32)
        assert torch.all(labels >= 0) and torch.all(labels < 10)

    def test_cifar_10_normalization(self):
        """Test that CIFAR-10 images are properly normalized."""
        train_loader, _ = load_cifar_10_other(batch_size=4)
        images, _ = next(iter(train_loader))

        # Check that images are normalized (approximately in range [-2, 2] for typical normalization)
        assert images.min() >= -3.0, f"Images too negative: {images.min()}"
        assert images.max() <= 3.0, f"Images too positive: {images.max()}"

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_cifar_10_batch_sizes(self, batch_size):
        """Test CIFAR-10 loading with different batch sizes."""
        train_loader, test_loader = load_cifar_10(batch_size=batch_size, perc_size=0.01)

        train_batch = next(iter(train_loader))
        images, labels = train_batch
        assert images.shape[0] <= batch_size  # May be smaller for last batch
        assert labels.shape[0] == images.shape[0]


class TestMNISTLoading:
    """Test MNIST data loading functions."""

    def test_load_mnist_basic(self):
        """Test basic MNIST loading functionality."""
        train_loader, test_loader, classes = load_mnist(batch_size=4)

        # Test classes
        assert len(classes) == 10
        assert classes == [str(i) for i in range(10)]

        # Test train loader
        assert isinstance(train_loader, DataLoader)
        train_batch = next(iter(train_loader))
        assert len(train_batch) == 2

        images, labels = train_batch
        assert images.shape[1:] == (1, 28, 28)  # MNIST dimensions
        assert images.shape[0] <= 4
        assert labels.shape[0] == images.shape[0]
        assert torch.all(labels >= 0) and torch.all(labels < 10)

    def test_mnist_normalization(self):
        """Test that MNIST images are properly normalized."""
        train_loader, _, _ = load_mnist(batch_size=4)
        images, _ = next(iter(train_loader))

        # MNIST uses different normalization than CIFAR-10
        assert images.min() >= -1.0, f"Images too negative: {images.min()}"
        assert images.max() <= 3.0, f"Images too positive: {images.max()}"


class TestDataConsistency:
    """Test consistency and properties across different datasets."""

    def test_data_types(self):
        """Test that all data loaders return correct tensor types."""
        # CIFAR-10
        train_loader, _ = load_cifar_10(batch_size=2, perc_size=0.01)
        images, labels = next(iter(train_loader))
        assert images.dtype == torch.float32
        assert labels.dtype == torch.int64

        # MNIST
        train_loader, _, _ = load_mnist(batch_size=2)
        images, labels = next(iter(train_loader))
        assert images.dtype == torch.float32
        assert labels.dtype == torch.int64

    @pytest.mark.slow
    def test_dataset_iteration(self):
        """Test that we can iterate through entire small datasets."""
        # Use very small subset for testing
        train_loader, _ = load_cifar_10(batch_size=4, perc_size=0.001)

        batch_count = 0
        total_samples = 0

        for images, labels in train_loader:
            batch_count += 1
            total_samples += images.shape[0]

            # Basic sanity checks
            assert images.ndim == 4
            assert labels.ndim == 1
            assert images.shape[0] == labels.shape[0]

            # Don't iterate too long in tests
            if batch_count >= 5:
                break

        assert batch_count > 0, "No batches found"
        assert total_samples > 0, "No samples found"
