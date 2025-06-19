import logging

import numpy as np
import pandas as pd
import torch
from ISLP import load_data
from ISLP.models import ModelSpec as MS
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class HittersDataModule(LightningDataModule):
    """Data module for Hitters dataset with MLOps best practices."""

    def __init__(
        self,
        dataset_name: str = "Hitters",
        test_size: float = 0.33,
        random_state: int = 42,
        batch_size: int = 32,
        num_workers: int = 4,
        standardize: bool = True,
        remove_outliers: bool = False,
        outlier_threshold: float = 3.0,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.random_state = random_state
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.standardize = standardize
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold

        # Will be set during setup
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = None
        self.input_size = None

    def prepare_data(self):
        """Download data if needed."""
        logger.info(f"Loading {self.dataset_name} dataset...")
        # ISLP handles the download automatically

    def setup(self, stage: str = None):
        """Set up train/test splits and preprocessing."""
        logger.info("Setting up data splits and preprocessing...")

        # Load data
        hitters = load_data(dataset=self.dataset_name).dropna()
        logger.info(f"Loaded dataset with {len(hitters)} samples")

        # Create feature matrix using ModelSpec
        model_spec = MS(hitters.columns.drop("Salary"), intercept=False)
        X = model_spec.fit_transform(hitters).to_numpy()
        y = hitters["Salary"].to_numpy()

        # Store feature names for interpretability
        self.feature_names = list(model_spec.column_names_.keys())
        self.input_size = X.shape[1]

        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target variable shape: {y.shape}")

        # Remove outliers if requested
        if self.remove_outliers:
            X, y = self._remove_outliers(X, y)
            logger.info(f"After outlier removal: {X.shape[0]} samples")

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        logger.info(f"Train set: {self.X_train.shape[0]} samples")
        logger.info(f"Test set: {self.X_test.shape[0]} samples")

        # Standardization
        if self.standardize:
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            logger.info("Applied standardization to features")

        # Convert to tensors
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32)

    def _remove_outliers(self, X, y):
        """Remove outliers based on z-score threshold."""
        z_scores = np.abs((y - y.mean()) / y.std())
        mask = z_scores < self.outlier_threshold
        return X[mask], y[mask]

    def train_dataloader(self):
        """Create training dataloader."""
        dataset = TensorDataset(self.X_train, self.y_train)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        """Create test dataloader."""
        dataset = TensorDataset(self.X_test, self.y_test)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        """Use test set as validation for now."""
        return self.test_dataloader()

    def get_feature_names(self):
        """Return feature names for interpretability."""
        return self.feature_names

    def get_data_stats(self):
        """Return dataset statistics for logging."""
        return {
            "n_features": self.input_size,
            "n_train": len(self.X_train),
            "n_test": len(self.X_test),
            "target_mean": float(self.y_train.mean()),
            "target_std": float(self.y_train.std()),
            "target_min": float(self.y_train.min()),
            "target_max": float(self.y_train.max()),
        }
