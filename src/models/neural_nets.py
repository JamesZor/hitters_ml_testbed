import logging
from typing import List, Optional

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

import wandb

logger = logging.getLogger(__name__)


class BaseRegressionModel(LightningModule):
    """Base class for regression models with common functionality."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        optimizer: str = "adam",
        weight_decay: float = 0.0001,
        lr_scheduler: Optional[dict] = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.weight_decay = weight_decay
        self.lr_scheduler_config = lr_scheduler

        # Metrics
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()

    def forward(self, x):
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)

        # Log metrics
        self.train_mae(y_hat, y)
        self.train_mse(y_hat, y)
        self.train_r2(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_mae", self.train_mae, on_step=False, on_epoch=True)
        self.log("train_mse", self.train_mse, on_step=False, on_epoch=True)
        self.log("train_r2", self.train_r2, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)

        # Log metrics
        self.val_mae(y_hat, y)
        self.val_mse(y_hat, y)
        self.val_r2(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae", self.val_mae, on_step=False, on_epoch=True)
        self.log("val_mse", self.val_mse, on_step=False, on_epoch=True)
        self.log("val_r2", self.val_r2, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)

        # Log metrics
        self.test_mae(y_hat, y)
        self.test_mse(y_hat, y)
        self.test_r2(y_hat, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_mae", self.test_mae, on_step=False, on_epoch=True)
        self.log("test_mse", self.test_mse, on_step=False, on_epoch=True)
        self.log("test_r2", self.test_r2, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Optimizer
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        elif self.optimizer_name.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        # Learning rate scheduler
        if self.lr_scheduler_config:
            scheduler_name = self.lr_scheduler_config.get("name", "reduce_on_plateau")

            if scheduler_name == "reduce_on_plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=self.lr_scheduler_config.get("factor", 0.5),
                    patience=self.lr_scheduler_config.get("patience", 5),
                    min_lr=self.lr_scheduler_config.get("min_lr", 1e-6),
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            elif scheduler_name == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.lr_scheduler_config.get("T_max", 50),
                    eta_min=self.lr_scheduler_config.get("min_lr", 1e-6),
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }

        return optimizer


class SimpleNN(BaseRegressionModel):
    """Simple feedforward neural network."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [50],
        dropout_rate: float = 0.4,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # Build network layers
        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))

            # Activation
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())
            elif activation.lower() == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

        logger.info(f"Created SimpleNN with architecture: {hidden_sizes}")

    def forward(self, x):
        """Forward pass."""
        return self.network(x).squeeze(-1)


class DeepNN(BaseRegressionModel):
    """Deeper neural network with batch normalization."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [64, 32, 16],
        dropout_rate: float = 0.3,
        activation: str = "relu",
        batch_norm: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # Build network layers
        layers = []
        prev_size = input_size

        # Hidden layers
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))

            # Batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            # Activation
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "leaky_relu":
                layers.append(nn.LeakyReLU(0.01))
            elif activation.lower() == "elu":
                layers.append(nn.ELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

        logger.info(
            f"Created DeepNN with architecture: {hidden_sizes}, batch_norm: {batch_norm}"
        )

    def forward(self, x):
        """Forward pass."""
        return self.network(x).squeeze(-1)


class ResidualNN(BaseRegressionModel):
    """Neural network with residual connections."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Residual blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_size, dropout_rate) for _ in range(num_blocks)]
        )

        # Output layer
        self.output = nn.Linear(hidden_size, 1)

        logger.info(
            f"Created ResidualNN with {num_blocks} blocks, hidden_size: {hidden_size}"
        )

    def forward(self, x):
        """Forward pass."""
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        return self.output(x).squeeze(-1)


class ResidualBlock(nn.Module):
    """Residual block for ResidualNN."""

    def __init__(self, hidden_size: int, dropout_rate: float = 0.2):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.BatchNorm1d(hidden_size)
        self.norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

    def forward(self, x):
        """Forward pass with residual connection."""
        residual = x

        out = self.layer1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.layer2(out)
        out = self.norm2(out)

        # Residual connection
        out += residual
        out = self.activation(out)

        return out
