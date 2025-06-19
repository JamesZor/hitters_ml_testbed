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
    def __init__(
        self,
        dataset_name: str = "Hitters",
        test_size: float = 0.33,
        random_state: int = 42,
        batch_size: int = 32,
        num_workers: int = 0,
        standardize: bool = True,
        standardize_categorical: bool = False,  # NEW: Control categorical standardization
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
        self.standardize_categorical = standardize_categorical
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold

        # Will be set during setup
        self.scaler = None
        self.numerical_columns = None
        self.categorical_columns = None
        self.feature_names = None

        # Will be set during setup
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.feature_names = None
        self.scaler = None
        self.input_size = None

    def setup(self, stage: str = None):
        """Setup data splits and preprocessing"""
        logger.info("Setting up data splits and preprocessing...")

        # Load the data
        logger.info(f"Loading {self.dataset_name} dataset...")
        data = load_data(dataset=self.dataset_name).dropna()

        logger.info(f"Loaded dataset with {len(data)} samples")

        # Create feature matrix using ModelSpec (handles categorical encoding)
        model_spec = MS(data.columns.drop("Salary"), intercept=False)
        X = model_spec.fit_transform(data)
        Y = data["Salary"].values

        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target variable shape: {Y.shape}")

        # Get feature names after ModelSpec transformation
        self.feature_names = X.columns.tolist()
        self.input_size = X.shape[1]

        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target variable shape: {Y.shape}")

        # Identify numerical vs categorical columns
        self._identify_column_types(X, data)

        # Convert to numpy for easier handling
        X_array = X.to_numpy()

        # Train-test split
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X_array, Y, test_size=self.test_size, random_state=self.random_state
        )

        logger.info(f"Train set: {len(self.X_train)} samples")
        logger.info(f"Test set: {len(self.X_test)} samples")

        # Remove outliers if specified
        if self.remove_outliers:
            self._remove_outliers()

        # Apply selective standardization
        if self.standardize:
            self._apply_selective_standardization()

        # Convert to tensors
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.Y_train = torch.tensor(self.Y_train, dtype=torch.float32)
        self.Y_test = torch.tensor(self.Y_test, dtype=torch.float32)

    def _identify_column_types(self, X_transformed, original_data):
        """Identify which columns are numerical vs categorical after ModelSpec transformation"""

        # Get original column info before transformation
        original_numerical = []
        original_categorical = []

        for col in original_data.columns:
            if col == "Salary":
                continue
            if original_data[col].dtype in ["object", "category"]:
                original_categorical.append(col)
            else:
                original_numerical.append(col)

        logger.info(f"Original numerical columns: {original_numerical}")
        logger.info(f"Original categorical columns: {original_categorical}")

        # After ModelSpec transformation, categorical columns become multiple binary columns
        # We need to map transformed column names back to their types
        self.numerical_columns = []
        self.categorical_columns = []

        for i, feature_name in enumerate(self.feature_names):
            # Check if this transformed column comes from a categorical variable
            is_categorical = False
            for cat_col in original_categorical:
                if feature_name.startswith(
                    cat_col + "["
                ):  # ModelSpec format: "League[A]"
                    is_categorical = True
                    break

            if is_categorical:
                self.categorical_columns.append(i)
            else:
                self.numerical_columns.append(i)

        logger.info(f"Transformed numerical column indices: {self.numerical_columns}")
        logger.info(
            f"Transformed categorical column indices: {self.categorical_columns}"
        )
        logger.info(
            f"Numerical features: {[self.feature_names[i] for i in self.numerical_columns]}"
        )
        logger.info(
            f"Categorical features: {[self.feature_names[i] for i in self.categorical_columns]}"
        )

    def _apply_selective_standardization(self):
        """Apply standardization only to numerical columns"""

        if not self.numerical_columns:
            logger.info("No numerical columns found - skipping standardization")
            return

        # Create a copy to avoid modifying original data
        X_train_scaled = self.X_train.copy()
        X_test_scaled = self.X_test.copy()

        # Extract numerical columns
        X_train_numerical = self.X_train[:, self.numerical_columns]
        X_test_numerical = self.X_test[:, self.numerical_columns]

        logger.info(f"Standardizing {len(self.numerical_columns)} numerical features")
        logger.info(
            f"Before standardization - numerical features mean: {X_train_numerical.mean(axis=0)[:5]}"
        )
        logger.info(
            f"Before standardization - numerical features std: {X_train_numerical.std(axis=0)[:5]}"
        )

        # Fit scaler only on numerical features
        self.scaler = StandardScaler()
        X_train_numerical_scaled = self.scaler.fit_transform(X_train_numerical)
        X_test_numerical_scaled = self.scaler.transform(X_test_numerical)

        logger.info(
            f"After standardization - numerical features mean: {X_train_numerical_scaled.mean(axis=0)[:5]}"
        )
        logger.info(
            f"After standardization - numerical features std: {X_train_numerical_scaled.std(axis=0)[:5]}"
        )

        # Replace numerical columns in the full feature matrix
        X_train_scaled[:, self.numerical_columns] = X_train_numerical_scaled
        X_test_scaled[:, self.numerical_columns] = X_test_numerical_scaled

        # Handle categorical columns
        if self.categorical_columns:
            if self.standardize_categorical:
                logger.info("Also standardizing categorical features (not recommended)")
                X_train_categorical = self.X_train[:, self.categorical_columns]
                X_test_categorical = self.X_test[:, self.categorical_columns]

                cat_scaler = StandardScaler()
                X_train_scaled[:, self.categorical_columns] = cat_scaler.fit_transform(
                    X_train_categorical
                )
                X_test_scaled[:, self.categorical_columns] = cat_scaler.transform(
                    X_test_categorical
                )
            else:
                logger.info(
                    f"Keeping {len(self.categorical_columns)} categorical features unchanged"
                )
                # Categorical columns remain unchanged (already in X_train_scaled/X_test_scaled)

        # Update the data
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled

    def _remove_outliers(self):
        """Remove outliers using Z-score method"""

        if not self.numerical_columns:
            logger.info("No numerical columns - skipping outlier removal")
            return

        # Calculate Z-scores for numerical columns only
        X_train_numerical = self.X_train[:, self.numerical_columns]
        z_scores = np.abs(
            (X_train_numerical - X_train_numerical.mean(axis=0))
            / X_train_numerical.std(axis=0)
        )

        # Find rows where any numerical feature is an outlier
        outlier_mask = (z_scores > self.outlier_threshold).any(axis=1)
        non_outlier_mask = ~outlier_mask

        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            logger.info(f"Removing {n_outliers} outliers from training set")
            self.X_train = self.X_train[non_outlier_mask]
            self.Y_train = self.Y_train[non_outlier_mask]
        else:
            logger.info("No outliers found")

    def train_dataloader(self):
        dataset = TensorDataset(self.X_train, self.Y_train)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        # Use test set as validation (like original)
        dataset = TensorDataset(self.X_test, self.Y_test)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        dataset = TensorDataset(self.X_test, self.Y_test)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def get_data_stats(self):
        """Return dataset statistics for logging."""
        return {
            "n_features": self.input_size,
            "n_train": len(self.X_train),
            "n_test": len(self.X_test),
            "target_mean": float(self.Y_train.mean()),
            "target_std": float(self.Y_train.std()),
            "target_min": float(self.Y_train.min()),
            "target_max": float(self.Y_train.max()),
        }

    def get_data_info(self):
        """Return information about the data"""
        return {
            "n_features": self.X_train.shape[1],
            "n_train": len(self.X_train),
            "n_test": len(self.X_test),
            "n_numerical_features": (
                len(self.numerical_columns) if self.numerical_columns else 0
            ),
            "n_categorical_features": (
                len(self.categorical_columns) if self.categorical_columns else 0
            ),
            "feature_names": self.feature_names,
            "numerical_features": (
                [self.feature_names[i] for i in self.numerical_columns]
                if self.numerical_columns
                else []
            ),
            "categorical_features": (
                [self.feature_names[i] for i in self.categorical_columns]
                if self.categorical_columns
                else []
            ),
            "target_mean": float(self.Y_train.mean()),
            "target_std": float(self.Y_train.std()),
            "target_min": float(self.Y_train.min()),
            "target_max": float(self.Y_train.max()),
        }
