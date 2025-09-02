# Torch-Playground: Computer Vision Experimentation Library

![CI](https://github.com/joeljosephjin/Torch-Playground/workflows/CI/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A PyTorch-based computer vision experimentation library focused on image classification research and model comparison.

## 🚀 Recent Improvements

This repository has been enhanced with modern development practices and infrastructure:

### ✅ Development Tools Added
- **Automated Code Formatting**: Black + isort for consistent code style
- **Linting**: Flake8 for code quality checks
- **Testing Framework**: Pytest with comprehensive test suite
- **CI/CD Pipeline**: GitHub Actions for automated testing
- **Pre-commit Hooks**: Automatic code quality checks
- **Modern Configuration**: pyproject.toml for Python packaging

### ✅ Code Quality Improvements
- Fixed import issues and bugs in existing code
- Added comprehensive test coverage for models, data loading, and utilities
- Improved error handling and type safety
- Better project structure and organization

## 📊 Features

### Supported Models
- **DenseNet**: DenseNet-40, DenseNet-100 with various growth rates
- **ResNet**: ResNet-18 implementation
- **Custom CNNs**: SimpleModel, AVModel for CIFAR-10/MNIST
- **Extensible**: Easy to add new architectures

### Datasets
- CIFAR-10 with data augmentation
- MNIST for grayscale classification
- Custom data loading utilities

### Experiment Tracking
- WandB integration for experiment logging
- Model checkpointing and resume functionality
- Reproducible experiments with seed setting

## 🛠 Setup & Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA support (optional but recommended)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/joeljosephjin/Torch-Playground.git
cd Torch-Playground

# Install dependencies
pip install -r requirements.txt

# For development
pip install -e .[dev]

# Setup pre-commit hooks (optional)
pre-commit install
```

## 🎯 Usage

### Training Models
```bash
# Train DenseNet on CIFAR-10
python main.py --epochs 300 --batch-size 64 --learning-rate 0.1 --model DenseNet3 --dataset cifar_10 --use-wandb

# Quick test run
make train-quick

# Train with different models
python main.py --model ResNet18 --dataset cifar_10 --epochs 50
python main.py --model SimpleModel --dataset mnist --epochs 10
```

### Development Commands
```bash
# Run tests
make test

# Format code
make format

# Lint code
make lint

# Type checking
make type-check

# All quality checks
make check-all
```

## 📈 Results

### DenseNet Performance on CIFAR-10

| Method (C10+) | Original Paper | Our Implementation |
|---------------|----------------|-------------------|
| ResNet        | 6.61%          | TBD               |
| DenseNet (k-12, d-40) | 5.24% | 5.6% [[R1]](https://wandb.ai/joeljosephjin/torch-cnn/runs/8rw76l3r) |
| DenseNet-BC (k-12, d-100) | 4.51% | TBD |
| DenseNet-BC (k-40, d-190) | 3.46% | TBD |

<img src="experiments/densenet_results.png" width="500" height="300">

### ResNet Results

[Paper](https://arxiv.org/pdf/1512.03385.pdf)

<img src="experiments/resnet_results_c10.png" width="300" height="300">

## 🧪 Testing

The project includes comprehensive tests covering:

- **Model Architecture Tests**: Forward pass, gradient flow, parameter counting
- **Data Loading Tests**: CIFAR-10, MNIST, batch consistency
- **Utility Tests**: Seed reproducibility, accuracy calculations
- **Integration Tests**: End-to-end training pipeline

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_models.py -v
pytest tests/test_data.py -v
pytest tests/test_utils.py -v

# Run with coverage (if pytest-cov installed)
pytest --cov=. --cov-report=html
```

## 📁 Project Structure

```
torch-playground/
├── data/                  # Data loading utilities
│   └── data.py
├── models/                # Model architectures
│   ├── models.py         # Basic models (SimpleModel, AVModel)
│   ├── densenet.py       # DenseNet implementation
│   ├── densenet3.py      # DenseNet variants
│   ├── resnet.py         # ResNet implementation
│   └── shufflenet.py     # ShuffleNet (WIP)
├── tests/                 # Test suite
│   ├── test_models.py
│   ├── test_data.py
│   └── test_utils.py
├── experiments/           # Experiment results and scripts
├── saved/                 # Model checkpoints
├── main.py               # Main training script
├── utils.py              # Utility functions
├── requirements.txt      # Dependencies
├── pyproject.toml        # Project configuration
├── Makefile             # Development commands
└── IMPROVEMENTS.md      # Detailed improvement suggestions
```

## 🔧 Configuration

The project supports flexible configuration through command-line arguments:

```bash
python main.py \
    --model DenseNet3 \
    --dataset cifar_10 \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.1 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --use-wandb \
    --save-as my_experiment
```

## 🎯 Research Goals

### Current Focus
1. **Reproduce Paper Results**: Achieve published accuracies for DenseNet/ResNet
2. **Ablation Studies**: Understand what makes models effective
3. **Hyperparameter Analysis**: Systematic tuning and analysis
4. **Model Comparison**: Fair benchmarking across architectures

### Research Questions
1. Does hyperparameter tuning on small dataset portions generalize?
2. Which optimizers work best for different datasets?
3. What are the benefits of fine-tuning vs. training from scratch?
4. How do different architectures compare with similar parameter counts?

## 🔮 Future Enhancements

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed roadmap including:

- **Advanced Training**: Mixed precision, early stopping, advanced optimizers
- **Extended Datasets**: ImageNet subsets, CIFAR-100, custom datasets
- **Analysis Tools**: Model visualization, performance profiling
- **Research Features**: Automated hyperparameter tuning, ensemble methods
- **Production**: Model serving, quantization, ONNX export

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run quality checks: `make check-all`
5. Submit a pull request

The project follows modern Python development practices with automated testing and code quality checks.

## 📚 References

- [DenseNet Paper](https://arxiv.org/pdf/1608.06993.pdf)
- [ResNet Paper](https://arxiv.org/pdf/1512.03385.pdf)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [WandB Documentation](https://docs.wandb.ai/)

## 📄 License

MIT License - feel free to use this code for research and educational purposes.

---

**Colab Notebook**: [Available here](https://colab.research.google.com/drive/1w5uEuyaX11vndVqPy5miFg8FOT892Ju1#scrollTo=YP1RRHgp1JO7)

*This library is designed to help researchers and students dive into computer vision model building, training, and research with modern development practices.*