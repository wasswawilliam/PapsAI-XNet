# src/training/trainer.py
"""
Two-stage training pipeline for PapsAI XNet.

Stage 1:
    Train CNN feature extractor using categorical cross-entropy.

Stage 2:
    Freeze CNN and train ANFIS on extracted 32-dimensional embeddings.

This mirrors the methodology described in the manuscript.
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.papsai_xnet import PapsAIXNet


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, validation_loss: float) -> bool:
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
            return True

        self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return False


class PapsAIXNetTrainer:
    def __init__(
        self,
        model: PapsAIXNet,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        config,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.config = config
        self.device = device

        self.checkpoint_dir = Path(config.checkpoints.directory)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()

    def train_cnn_stage(self) -> Dict[str, list]:
        """
        Stage 1: Train CNN backbone and ANFIS classifier end-to-end
        using cross-entropy loss.

        The best model is selected using validation loss.
        """

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            betas=(self.config.training.beta1, self.config.training.beta2),
            weight_decay=self.config.training.weight_decay,
        )

        early_stopping = EarlyStopping(
            patience=self.config.early_stopping.patience,
            min_delta=self.config.early_stopping.min_delta,
        )

        history = {
            "train_loss": [],
            "validation_loss": [],
            "train_accuracy": [],
            "validation_accuracy": [],
        }

        for epoch in range(self.config.training.epochs):
            train_loss, train_accuracy = self._train_one_epoch(optimizer)
            validation_loss, validation_accuracy = self._validate()

            history["train_loss"].append(train_loss)
            history["validation_loss"].append(validation_loss)
            history["train_accuracy"].append(train_accuracy)
            history["validation_accuracy"].append(validation_accuracy)

            print(
                f"Epoch [{epoch + 1}/{self.config.training.epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Train Acc: {train_accuracy:.4f} "
                f"Val Loss: {validation_loss:.4f} "
                f"Val Acc: {validation_accuracy:.4f}"
            )

            improved = early_stopping.step(validation_loss)

            if improved:
                self.save_checkpoint(self.config.checkpoints.best_model)

            if early_stopping.should_stop:
                print("Early stopping triggered.")
                break

        self.save_checkpoint(self.config.checkpoints.last_model)

        return history

    def train_anfis_stage(self) -> Dict[str, list]:
        """
        Stage 2: Freeze CNN and optimize ANFIS parameters.

        This follows the manuscript description:
        CNN weights are frozen and extracted embeddings are used as fixed
        inputs to the ANFIS module.
        """

        self.model.freeze_cnn()

        optimizer = torch.optim.Adam(
            self.model.anfis.parameters(),
            lr=self.config.anfis.learning_rate,
        )

        history = {
            "anfis_train_loss": [],
            "anfis_validation_loss": [],
            "parameter_update_norm": [],
        }

        previous_parameters = self._flatten_anfis_parameters().detach().clone()

        for epoch in range(self.config.training.epochs):
            train_loss, _ = self._train_one_epoch(optimizer)
            validation_loss, _ = self._validate()

            current_parameters = self._flatten_anfis_parameters().detach().clone()
            update_norm = torch.norm(current_parameters - previous_parameters, p=2).item()
            previous_parameters = current_parameters

            history["anfis_train_loss"].append(train_loss)
            history["anfis_validation_loss"].append(validation_loss)
            history["parameter_update_norm"].append(update_norm)

            print(
                f"ANFIS Epoch [{epoch + 1}/{self.config.training.epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Val Loss: {validation_loss:.4f} "
                f"Update Norm: {update_norm:.6f}"
            )

            if update_norm < self.config.anfis.convergence_threshold:
                print("ANFIS convergence criterion reached.")
                break

        self.save_checkpoint("papsai_xnet_anfis_trained.pth")

        return history

    def _train_one_epoch(self, optimizer) -> Tuple[float, float]:
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        average_loss = total_loss / total
        accuracy = correct / total

        return average_loss, accuracy

    @torch.no_grad()
    def _validate(self) -> Tuple[float, float]:
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(self.validation_loader, desc="Validation", leave=False)

        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        average_loss = total_loss / total
        accuracy = correct / total

        return average_loss, accuracy

    def _flatten_anfis_parameters(self) -> torch.Tensor:
        parameters = [
            parameter.view(-1)
            for parameter in self.model.anfis.parameters()
            if parameter.requires_grad
        ]

        return torch.cat(parameters)

    def save_checkpoint(self, filename: str) -> None:
        checkpoint_path = self.checkpoint_dir / filename

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config.to_dict() if hasattr(self.config, "to_dict") else None,
            },
            checkpoint_path,
        )

        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Checkpoint loaded: {checkpoint_path}")
