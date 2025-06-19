import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.data.dataloader import HittersDataModule
from src.models.neural_nets import DeepNN, ResidualNN, SimpleNN
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def load_model_from_experiment(experiment_dir: str):
    """Load model from experiment directory (handles both checkpoints and final model)."""

    experiment_path = Path(experiment_dir)

    # Try different config file locations
    config_paths = [
        experiment_path / "config.yaml",
        experiment_path / ".hydra" / "config.yaml",
        experiment_path / "hydra_config.yaml",
    ]

    config_path = None
    for path in config_paths:
        if path.exists():
            config_path = path
            break

    if config_path is None:
        raise FileNotFoundError(
            f"Config file not found in {experiment_path}. Tried: {[str(p) for p in config_paths]}"
        )

    # Load config
    cfg = OmegaConf.load(config_path)

    # Initialize data module
    data_config = OmegaConf.to_container(cfg.data, resolve=True)
    data_config.pop("_target_", None)
    data_module = HittersDataModule(**data_config)
    data_module.setup()

    # Get model parameters
    model_name = cfg.model._target_.split(".")[-1]
    model_params = OmegaConf.to_container(cfg.model, resolve=True)
    model_params.pop("_target_", None)
    model_params["input_size"] = data_module.get_data_stats()["n_features"]

    # Create model
    if model_name == "SimpleNN":
        model = SimpleNN(**model_params)
    elif model_name == "DeepNN":
        model = DeepNN(**model_params)
    elif model_name == "ResidualNN":
        model = ResidualNN(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Try to load checkpoint first, then final model
    checkpoint_dir = experiment_path / "checkpoints"
    final_model_path = experiment_path / "final_model.pt"

    if checkpoint_dir.exists() and any(checkpoint_dir.glob("*.ckpt")):
        # Load best checkpoint
        checkpoint_files = list(checkpoint_dir.glob("best-*.ckpt"))
        if checkpoint_files:
            checkpoint_path = checkpoint_files[0]  # Take first best checkpoint
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])
        else:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    elif final_model_path.exists():
        # Load final model
        logger.info(f"Loading final model: {final_model_path}")
        model.load_state_dict(torch.load(final_model_path, map_location="cpu"))
    else:
        raise FileNotFoundError(f"No model files found in {experiment_path}")

    model.eval()
    return model, data_module, cfg


def evaluate_experiment(experiment_dir: str, output_dir: str = None):
    """Evaluate a trained model from experiment directory."""

    setup_logging()
    logger.info(f"Evaluating experiment: {experiment_dir}")

    # Load model and data
    model, data_module, cfg = load_model_from_experiment(experiment_dir)

    # Make predictions
    test_loader = data_module.test_dataloader()

    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            y_pred = model(x)

            y_true_list.append(y.numpy())
            y_pred_list.append(y_pred.numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    # Calculate metrics
    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))

    # Print results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Model: {cfg.model._target_.split('.')[-1]}")
    logger.info(f"Architecture: {cfg.model.get('hidden_sizes', 'N/A')}")
    logger.info(f"Test samples: {len(y_true)}")
    logger.info(f"Mean Absolute Error: {mae:.2f}")
    logger.info(f"Root Mean Square Error: {rmse:.2f}")
    logger.info(f"R² Score: {r2:.4f}")
    logger.info(f"Actual salary range: ${y_true.min():.0f} - ${y_true.max():.0f}")
    logger.info(f"Predicted salary range: ${y_pred.min():.0f} - ${y_pred.max():.0f}")

    # Performance assessment
    if r2 < 0:
        logger.warning(
            "⚠️  POOR PERFORMANCE: Negative R² means model is worse than predicting the mean!"
        )
    elif r2 < 0.3:
        logger.warning("⚠️  WEAK PERFORMANCE: R² < 0.3 indicates poor model fit")
    elif r2 < 0.6:
        logger.info("✅ MODERATE PERFORMANCE: R² indicates reasonable model fit")
    else:
        logger.info("🎯 GOOD PERFORMANCE: High R² indicates good model fit")

    # Create output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(experiment_dir) / "evaluation"
        output_path.mkdir(exist_ok=True)

    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Predictions vs Actual
    ax1.scatter(y_true, y_pred, alpha=0.6, s=30)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot(
        [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction"
    )
    ax1.set_xlabel("Actual Salary ($1000s)")
    ax1.set_ylabel("Predicted Salary ($1000s)")
    ax1.set_title(f"Predicted vs Actual (R² = {r2:.3f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Residuals
    residuals = y_pred - y_true
    ax2.scatter(y_pred, residuals, alpha=0.6, s=30)
    ax2.axhline(y=0, color="r", linestyle="--", lw=2)
    ax2.set_xlabel("Predicted Salary ($1000s)")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residuals vs Predicted")
    ax2.grid(True, alpha=0.3)

    # 3. Error distribution
    ax3.hist(np.abs(residuals), bins=20, alpha=0.7, edgecolor="black")
    ax3.axvline(mae, color="red", linestyle="--", linewidth=2, label=f"MAE = {mae:.1f}")
    ax3.set_xlabel("Absolute Error")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Distribution of Absolute Errors")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Sample predictions
    n_samples = min(10, len(y_true))
    sample_idx = np.random.choice(len(y_true), n_samples, replace=False)
    x_pos = range(n_samples)

    ax4.bar(
        [x - 0.2 for x in x_pos],
        y_true[sample_idx],
        width=0.4,
        label="Actual",
        alpha=0.7,
    )
    ax4.bar(
        [x + 0.2 for x in x_pos],
        y_pred[sample_idx],
        width=0.4,
        label="Predicted",
        alpha=0.7,
    )
    ax4.set_xlabel("Sample Index")
    ax4.set_ylabel("Salary ($1000s)")
    ax4.set_title("Sample Predictions")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_path / "evaluation_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Save detailed results
    results_df = pd.DataFrame(
        {
            "actual": y_true,
            "predicted": y_pred,
            "error": residuals,
            "abs_error": np.abs(residuals),
            "relative_error": np.abs(residuals) / y_true * 100,
        }
    )

    results_df.to_csv(output_path / "detailed_results.csv", index=False)

    # Save summary
    summary = {
        "model": cfg.model._target_.split(".")[-1],
        "architecture": cfg.model.get("hidden_sizes", "N/A"),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "n_samples": len(y_true),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    OmegaConf.save(summary, output_path / "evaluation_summary.yaml")

    logger.info(f"Evaluation complete! Results saved to: {output_path}")
    logger.info(f"Plots saved to: {plot_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("experiment_dir", help="Path to experiment directory")
    parser.add_argument("--output_dir", help="Output directory for results")

    args = parser.parse_args()
    evaluate_experiment(args.experiment_dir, args.output_dir)
