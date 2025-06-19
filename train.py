import logging
import os
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

import wandb

# Import our modules
from src.data.dataloader import HittersDataModule
from src.models.neural_nets import DeepNN, ResidualNN, SimpleNN
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def instantiate_model(model_config: DictConfig, input_size: int):
    """Instantiate model from config."""
    model_name = model_config._target_.split(".")[-1]

    # Remove _target_ from config for model instantiation
    model_params = OmegaConf.to_container(model_config, resolve=True)
    model_params.pop("_target_", None)
    model_params["input_size"] = input_size

    if model_name == "SimpleNN":
        return SimpleNN(**model_params)
    elif model_name == "DeepNN":
        return DeepNN(**model_params)
    elif model_name == "ResidualNN":
        return ResidualNN(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")


@hydra.main(version_base=None, config_path="configs", config_name="experiment")
def train(cfg: DictConfig) -> float:
    """Main training function."""

    # Setup logging
    setup_logging()
    logger.info("Starting training...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seeds
    pl.seed_everything(cfg.seed, workers=True)

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)

    # Initialize data module
    logger.info("Initializing data module...")
    data_config = OmegaConf.to_container(cfg.data, resolve=True)
    data_config.pop("_target_", None)

    data_module = HittersDataModule(**data_config)
    data_module.setup()

    # Log data statistics
    data_stats = data_module.get_data_stats()
    logger.info(f"Data statistics: {data_stats}")

    # Initialize model
    logger.info("Initializing model...")
    model = instantiate_model(cfg.model, data_stats["n_features"])

    # Log model architecture
    logger.info(f"Model: {model}")
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize Weights & Biases logger
    wandb_logger = None
    if cfg.wandb.mode != "disabled":
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"{cfg.experiment.name}_{cfg.model._target_.split('.')[-1]}",
            tags=cfg.experiment.tags,
            notes=cfg.experiment.notes,
            mode=cfg.wandb.mode,
            save_dir=str(output_dir),
        )

        # Log configuration and data stats
        wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
        wandb_logger.experiment.config.update(data_stats)

        # Log model architecture
        wandb_logger.watch(model, log="all", log_freq=100)

    # Setup callbacks
    callbacks = []

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-{epoch:02d}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=cfg.training.save_top_k,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=cfg.training.patience,
        min_delta=cfg.training.min_delta,
        verbose=True,
    )
    callbacks.append(early_stopping)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train the model
    logger.info("Starting training...")
    trainer.fit(model, data_module)

    # Test the model
    logger.info("Testing model...")
    test_results = trainer.test(model, data_module, ckpt_path="best")

    # Log final results
    best_val_loss = checkpoint_callback.best_model_score.item()
    test_loss = test_results[0]["test_loss"]
    test_mae = test_results[0]["test_mae"]
    test_r2 = test_results[0]["test_r2"]

    logger.info(f"Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Test loss: {test_loss:.4f}")
    logger.info(f"Test MAE: {test_mae:.4f}")
    logger.info(f"Test R²: {test_r2:.4f}")

    # Save final model
    model_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    # Log to wandb if enabled
    if wandb_logger:
        wandb_logger.experiment.log(
            {
                "final/best_val_loss": best_val_loss,
                "final/test_loss": test_loss,
                "final/test_mae": test_mae,
                "final/test_r2": test_r2,
            }
        )

        # Save model artifact
        artifact = wandb.Artifact(
            name=f"model-{wandb_logger.experiment.id}",
            type="model",
            description=f"Trained {cfg.model._target_.split('.')[-1]} model",
        )
        artifact.add_file(str(model_path))
        wandb_logger.experiment.log_artifact(artifact)

        wandb.finish()

    return test_mae  # Return main metric for hyperparameter optimization


if __name__ == "__main__":
    train()


# Example usage commands:
"""
# Basic training with default config
python train.py

# Use different model
python train.py model=deep_nn

# Override specific parameters
python train.py model.hidden_sizes=[128,64,32] training.max_epochs=100

# Change multiple configs
python train.py model=deep_nn data.batch_size=64 training.learning_rate=0.01

# Run with specific experiment name
python train.py experiment.name="deep_network_experiment"

# Disable wandb logging
python train.py wandb.mode=disabled

# Run hyperparameter sweep (see sweep.py for sweep config)
python train.py --multirun model=simple_nn,deep_nn model.learning_rate=0.001,0.01,0.1
"""
