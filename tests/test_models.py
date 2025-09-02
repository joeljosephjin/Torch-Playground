"""Tests for model architectures."""

import pytest
import torch

from models.densenet3 import DenseNet3
from models.models import AVModel, SimpleMNIST, SimpleModel
from models.resnet import ResNet18


class TestSimpleModel:
    """Test SimpleModel architecture."""

    def test_simple_model_forward(self, sample_cifar_batch, device):
        """Test SimpleModel forward pass."""
        model = SimpleModel().to(device)
        images, _ = sample_cifar_batch
        images = images.to(device)

        output = model(images)

        assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_simple_model_parameters(self):
        """Test SimpleModel has trainable parameters."""
        model = SimpleModel()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params > 0, "Model has no parameters"
        assert trainable_params == total_params, "Some parameters are not trainable"


class TestAVModel:
    """Test AVModel architecture."""

    def test_av_model_forward(self, sample_cifar_batch, device):
        """Test AVModel forward pass."""
        model = AVModel().to(device)
        images, _ = sample_cifar_batch
        images = images.to(device)

        output = model(images)

        assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"


class TestSimpleMNIST:
    """Test SimpleMNIST architecture."""

    def test_simple_mnist_forward(self, sample_mnist_batch, device):
        """Test SimpleMNIST forward pass."""
        model = SimpleMNIST().to(device)
        images, _ = sample_mnist_batch
        images = images.to(device)

        output = model(images)

        assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"


class TestDenseNet3:
    """Test DenseNet3 architecture."""

    @pytest.mark.parametrize(
        "depth,growth_rate",
        [
            (40, 12),
            (100, 12),
            (40, 24),
        ],
    )
    def test_densenet3_forward(self, sample_cifar_batch, device, depth, growth_rate):
        """Test DenseNet3 forward pass with different configurations."""
        model = DenseNet3(depth=depth, num_classes=10, growth_rate=growth_rate).to(
            device
        )
        images, _ = sample_cifar_batch
        images = images.to(device)

        output = model(images)

        assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_densenet3_gradient_flow(self, sample_cifar_batch, device):
        """Test that gradients flow properly through DenseNet3."""
        model = DenseNet3(depth=40, num_classes=10, growth_rate=12).to(device)
        images, labels = sample_cifar_batch
        images, labels = images.to(device), labels.to(device)

        criterion = torch.nn.CrossEntropyLoss()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()

        # Check that gradients are computed
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients, "No gradients computed"


class TestResNet18:
    """Test ResNet18 architecture."""

    def test_resnet18_forward(self, sample_cifar_batch, device):
        """Test ResNet18 forward pass."""
        model = ResNet18().to(device)
        images, _ = sample_cifar_batch
        images = images.to(device)

        output = model(images)

        assert output.shape == (4, 10), f"Expected shape (4, 10), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"


class TestModelConsistency:
    """Test consistency across different models."""

    @pytest.mark.parametrize("model_class", [SimpleModel, AVModel, DenseNet3, ResNet18])
    def test_model_reproducibility(
        self, model_class, sample_cifar_batch, random_seed, device
    ):
        """Test that models produce consistent outputs with fixed seed."""
        if model_class == DenseNet3:
            model1 = model_class(depth=40, num_classes=10, growth_rate=12).to(device)
            model2 = model_class(depth=40, num_classes=10, growth_rate=12).to(device)
        else:
            model1 = model_class().to(device)
            model2 = model_class().to(device)

        # Copy parameters from model1 to model2
        model2.load_state_dict(model1.state_dict())

        images, _ = sample_cifar_batch
        images = images.to(device)

        # Set both models to eval mode
        model1.eval()
        model2.eval()

        with torch.no_grad():
            output1 = model1(images)
            output2 = model2(images)

        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-8)
