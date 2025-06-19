# src/visualization/plots.py
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import wandb


def plot_training_curves(
    trainer, save_path: Optional[str] = None, log_to_wandb: bool = False
):
    """Plot training and validation curves."""

    # Get metrics from trainer
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []

    for logged_metrics in trainer.logged_metrics_history:
        if "train_loss_epoch" in logged_metrics:
            train_losses.append(logged_metrics["train_loss_epoch"])
        if "val_loss" in logged_metrics:
            val_losses.append(logged_metrics["val_loss"])
        if "train_mae" in logged_metrics:
            train_maes.append(logged_metrics["train_mae"])
        if "val_mae" in logged_metrics:
            val_maes.append(logged_metrics["val_mae"])

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, label="Training Loss", color="blue")
    ax1.plot(epochs, val_losses, label="Validation Loss", color="red")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot MAE
    ax2.plot(epochs, train_maes, label="Training MAE", color="blue")
    ax2.plot(epochs, val_maes, label="Validation MAE", color="red")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean Absolute Error")
    ax2.set_title("Training and Validation MAE")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if log_to_wandb:
        wandb.log({"training_curves": wandb.Image(fig)})

    return fig


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    log_to_wandb: bool = False,
):
    """Plot predictions vs actual values."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.6, s=30)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot(
        [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction"
    )

    ax1.set_xlabel("Actual Salary ($1000s)")
    ax1.set_ylabel("Predicted Salary ($1000s)")
    ax1.set_title("Predicted vs Actual Salaries")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Residuals plot
    residuals = y_pred - y_true
    ax2.scatter(y_pred, residuals, alpha=0.6, s=30)
    ax2.axhline(y=0, color="r", linestyle="--", lw=2)
    ax2.set_xlabel("Predicted Salary ($1000s)")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residuals vs Predicted Values")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if log_to_wandb:
        wandb.log({"predictions": wandb.Image(fig)})

    return fig
