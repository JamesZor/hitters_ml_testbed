# Baseball Salary Prediction - MLOps Project

This project demonstrates MLOps best practices for a baseball salary prediction task using PyTorch, PyTorch Lightning, Weights & Biases, and Hydra.

## Features

- 🔧 **Modular Architecture**: Separate modules for data, models, training, and utilities
- ⚙️ **Configuration Management**: Hydra-based configs for easy experimentation
- 📊 **Experiment Tracking**: Weights & Biases integration for experiment monitoring
- 🏗️ **Multiple Model Architectures**: Simple NN, Deep NN, and Residual NN
- 🔄 **Hyperparameter Optimization**: Automated hyperparameter sweeps
- 📈 **Comprehensive Evaluation**: Detailed model analysis and visualization
- 🧪 **Reproducible**: Deterministic training with proper seed management

## Setup

1. **Create project structure:**
```bash
git clone <your-repo>
cd hitters_ml_project
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
# or
pip install -e .
```

3. **Setup Weights & Biases:**
```bash
wandb login
```

## Usage

### Basic Training

```bash
# Train with default configuration
python train.py

# Use different model
python train.py model=deep_nn

# Override parameters
python train.py model.learning_rate=0.01 training.max_epochs=100

# Disable wandb logging
python train.py wandb.mode=disabled
```

### Hyperparameter Sweeps

```bash
# Initialize sweep
wandb sweep sweep.yaml

# Run sweep agent
wandb agent <sweep-id>
```

### Model Evaluation

```bash
# Evaluate trained model
python evaluate.py experiments/2024-01-15_10-30-45/checkpoints/best-epoch=42-val_loss=0.123.ckpt --output_dir results/
```

### Advanced Usage

```bash
# Multi-run experiments
python train.py --multirun model=simple_nn,deep_nn model.learning_rate=0.001,0.01

# Experiment with different data configurations
python train.py data.batch_size=64 data.standardize=false

# Custom experiment naming
python train.py experiment.name="my_custom_experiment"
```

## Project Structure

```
├── configs/          # Hydra configuration files
├── src/             # Source code
│   ├── data/        # Data loading and preprocessing
│   ├── models/      # Model architectures
│   ├── training/    # Training utilities
│   ├── utils/       # General utilities
│   └── visualization/ # Plotting functions
├── experiments/     # Saved experiment outputs
├── models/         # Saved model checkpoints
└── notebooks/      # Jupyter notebooks
```

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/experiment.yaml`: Main experiment configuration
- `configs/model/`: Model architecture configurations
- `configs/data/`: Data loading configurations
- `configs/training/`: Training hyperparameters

## MLOps Features

- **Experiment Tracking**: All runs logged to Weights & Biases
- **Model Versioning**: Automatic model checkpointing and versioning
- **Reproducibility**: Deterministic training with seed management
- **Configuration Management**: Easy parameter tuning via YAML configs
- **Automated Evaluation**: Comprehensive model evaluation pipeline
- **Hyperparameter Optimization**: Bayesian optimization with W&B sweeps

## Results
