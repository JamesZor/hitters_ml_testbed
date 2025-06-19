# evaluate.py - Evaluation script
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
from src.visualization.plots import plot_predictions

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path: str, config_path: str):
    """Load trained model from checkpoint."""

    # Load config
    cfg = OmegaConf.load(config_path)

    # Initialize data module to get input size
    data_config = OmegaConf.to_container(cfg.data, resolve=True)
    data_config.pop("_target_", None)
    data_module = HittersDataModule(**data_config)
    data_module.setup()

    # Get model class
    model_name = cfg.model._target_.split(".")[-1]
    model_params = OmegaConf.to_container(cfg.model, resolve=True)
    model_params.pop("_target_", None)
    model_params["input_size"] = data_module.get_data_stats()["n_features"]

    if model_name == "SimpleNN":
        model = SimpleNN(**model_params)
    elif model_name == "DeepNN":
        model = DeepNN(**model_params)
    elif model_name == "ResidualNN":
        model = ResidualNN(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model, data_module


def evaluate_model(model_path: str, output_dir: str = None):
    """Evaluate a trained model."""

    setup_logging()
    logger.info(f"Evaluating model: {model_path}")

    # Find config file
    model_dir = Path(model_path).parent
    config_path = model_dir / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load model and data
    model, data_module = load_model_from_checkpoint(model_path, config_path)

    # Make predictions
    model.eval()
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
    logger.info(f"Test samples: {len(y_true)}")
    logger.info(f"Mean Absolute Error: {mae:.2f}")
    logger.info(f"Root Mean Square Error: {rmse:.2f}")
    logger.info(f"R² Score: {r2:.4f}")
    logger.info(f"Actual salary range: ${y_true.min():.0f} - ${y_true.max():.0f}")
    logger.info(f"Predicted salary range: ${y_pred.min():.0f} - ${y_pred.max():.0f}")

    # Create detailed analysis
    results_df = pd.DataFrame(
        {
            "actual": y_true,
            "predicted": y_pred,
            "error": y_pred - y_true,
            "abs_error": np.abs(y_pred - y_true),
            "relative_error": np.abs(y_pred - y_true) / y_true * 100,
        }
    )

    # Show worst and best predictions
    worst_idx = results_df["abs_error"].idxmax()
    best_idx = results_df["abs_error"].idxmin()

    logger.info(f"\nWorst prediction:")
    logger.info(f"  Actual: ${results_df.loc[worst_idx, 'actual']:.0f}")
    logger.info(f"  Predicted: ${results_df.loc[worst_idx, 'predicted']:.0f}")
    logger.info(f"  Error: ${results_df.loc[worst_idx, 'error']:.0f}")

    logger.info(f"\nBest prediction:")
    logger.info(f"  Actual: ${results_df.loc[best_idx, 'actual']:.0f}")
    logger.info(f"  Predicted: ${results_df.loc[best_idx, 'predicted']:.0f}")
    logger.info(f"  Error: ${results_df.loc[best_idx, 'error']:.0f}")

    # Create plots
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Predictions plot
        plot_predictions(y_true, y_pred, save_path=output_path / "predictions.png")

        # Save detailed results
        results_df.to_csv(output_path / "detailed_results.csv", index=False)

        logger.info(f"Results saved to: {output_path}")

    return {"mae": mae, "rmse": rmse, "r2": r2, "results_df": results_df}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("model_path", help="Path to model checkpoint")
    parser.add_argument("--output_dir", help="Output directory for results")

    args = parser.parse_args()
    evaluate_model(args.model_path, args.output_dir)
