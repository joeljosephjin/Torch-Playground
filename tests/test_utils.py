"""Tests for utility functions."""

import os

import numpy as np
import pytest
import torch

from utils import accuracy_densenet, set_seed


class TestSetSeed:
    """Test seed setting functionality."""

    def test_set_seed_reproducibility(self):
        """Test that setting seed produces reproducible results."""
        # Set seed and generate random numbers
        set_seed(42)
        torch_rand1 = torch.rand(5)
        np_rand1 = np.random.rand(5)

        # Set seed again and generate same random numbers
        set_seed(42)
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)

        # Check reproducibility
        torch.testing.assert_close(torch_rand1, torch_rand2)
        np.testing.assert_array_equal(np_rand1, np_rand2)

    def test_set_seed_different_seeds(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        torch_rand1 = torch.rand(5)

        set_seed(123)
        torch_rand2 = torch.rand(5)

        # Should not be equal
        assert not torch.allclose(torch_rand1, torch_rand2)

    def test_set_seed_environment_variable(self):
        """Test that PYTHONHASHSEED is set correctly."""
        original_hashseed = os.environ.get("PYTHONHASHSEED")

        set_seed(42)
        assert os.environ.get("PYTHONHASHSEED") == "42"

        # Restore original value if it existed
        if original_hashseed is not None:
            os.environ["PYTHONHASHSEED"] = original_hashseed


class TestAccuracyDenseNet:
    """Test accuracy calculation function."""

    def test_accuracy_densenet_perfect_prediction(self):
        """Test accuracy calculation with perfect predictions."""
        batch_size = 4
        num_classes = 10

        # Create perfect predictions
        targets = torch.tensor([0, 1, 2, 3])
        logits = torch.zeros(batch_size, num_classes)

        # Set highest logit for correct class
        for i, target in enumerate(targets):
            logits[i, target] = 10.0  # High confidence for correct class

        accuracy = accuracy_densenet(logits, targets, topk=(1,))[0]
        assert (
            accuracy.item() == 100.0
        ), f"Expected 100% accuracy, got {accuracy.item()}"

    def test_accuracy_densenet_random_prediction(self):
        """Test accuracy calculation with random predictions."""
        batch_size = 100
        num_classes = 10

        targets = torch.randint(0, num_classes, (batch_size,))
        logits = torch.randn(batch_size, num_classes)

        accuracy = accuracy_densenet(logits, targets, topk=(1,))[0]

        # Random accuracy should be around 10% for 10 classes, but allow wide range
        assert 0.0 <= accuracy.item() <= 100.0
        # For random predictions, unlikely to get very high accuracy
        assert accuracy.item() <= 50.0  # Very lenient upper bound

    def test_accuracy_densenet_worst_prediction(self):
        """Test accuracy calculation with worst possible predictions."""
        batch_size = 4
        num_classes = 10

        targets = torch.tensor([0, 1, 2, 3])
        logits = torch.zeros(batch_size, num_classes)

        # Set highest logit for wrong class
        for i, target in enumerate(targets):
            wrong_class = (target + 1) % num_classes
            logits[i, wrong_class] = 10.0

        accuracy = accuracy_densenet(logits, targets, topk=(1,))[0]
        assert accuracy.item() == 0.0, f"Expected 0% accuracy, got {accuracy.item()}"

    def test_accuracy_densenet_top5(self):
        """Test top-5 accuracy calculation."""
        batch_size = 4
        num_classes = 10

        targets = torch.tensor([0, 1, 2, 3])
        logits = torch.randn(batch_size, num_classes)

        # Ensure correct class is in top-5 by setting high values
        for i, target in enumerate(targets):
            logits[i, target] = 5.0  # High but not necessarily highest

        top1_acc, top5_acc = accuracy_densenet(logits, targets, topk=(1, 5))

        assert 0.0 <= top1_acc.item() <= 100.0
        assert 0.0 <= top5_acc.item() <= 100.0
        assert top5_acc.item() >= top1_acc.item()  # Top-5 should be >= top-1

    def test_accuracy_densenet_single_sample(self):
        """Test accuracy calculation with single sample."""
        targets = torch.tensor([5])
        logits = torch.zeros(1, 10)
        logits[0, 5] = 10.0  # Correct prediction

        accuracy = accuracy_densenet(logits, targets, topk=(1,))[0]
        assert accuracy.item() == 100.0

    def test_accuracy_densenet_data_types(self):
        """Test that accuracy function handles different tensor types correctly."""
        targets = torch.tensor([0, 1, 2, 3])
        logits = torch.randn(4, 10)

        # Test with different tensor types
        accuracy = accuracy_densenet(logits, targets, topk=(1,))[0]
        assert isinstance(accuracy, torch.Tensor)
        assert accuracy.dtype == torch.float32

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
    def test_accuracy_densenet_different_batch_sizes(self, batch_size):
        """Test accuracy calculation with different batch sizes."""
        targets = torch.randint(0, 10, (batch_size,))
        logits = torch.randn(batch_size, 10)

        accuracy = accuracy_densenet(logits, targets, topk=(1,))[0]

        assert 0.0 <= accuracy.item() <= 100.0
        assert isinstance(accuracy, torch.Tensor)


class TestUtilsIntegration:
    """Integration tests for utility functions."""

    def test_seed_accuracy_integration(self):
        """Test that seeded operations produce consistent accuracy calculations."""
        set_seed(42)
        targets1 = torch.randint(0, 10, (20,))
        logits1 = torch.randn(20, 10)
        accuracy1 = accuracy_densenet(logits1, targets1, topk=(1,))[0]

        set_seed(42)
        targets2 = torch.randint(0, 10, (20,))
        logits2 = torch.randn(20, 10)
        accuracy2 = accuracy_densenet(logits2, targets2, topk=(1,))[0]

        torch.testing.assert_close(targets1, targets2)
        torch.testing.assert_close(logits1, logits2)
        torch.testing.assert_close(accuracy1, accuracy2)
