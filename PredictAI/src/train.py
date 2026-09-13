"""
train.py

Training pipeline for Spacecraft Intelligence.

Responsibilities
----------------
✔ Load telemetry channel
✔ Preprocess data
✔ Create sliding windows
✔ Create DataLoader
✔ Train LSTM Autoencoder
✔ Save best model
✔ Save scaler
✔ Save training history
✔ Save reconstruction errors
"""

from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.dataset import NASADataset
from src.preprocessing import (
    DataPreprocessor,
    WindowGenerator,
)

from src.model import LSTMAutoencoder
from src.dataset import TelemetryDataset


class Trainer:

    def __init__(self, channel: str):

        self.channel = channel
        self.device = Config.DEVICE

        print("=" * 60)
        print("SPACECRAFT INTELLIGENCE")
        print("Training LSTM Autoencoder")
        print("=" * 60)

        print(f"Channel : {channel}")
        print(f"Device  : {self.device}")

        self.dataset = NASADataset()

        self.preprocessor = DataPreprocessor()

        self.window_generator = WindowGenerator()

    def prepare_data(self):

        train, _, _ = self.dataset.load_channel(self.channel)

        train = self.preprocessor.fit_transform(train)

        self.preprocessor.save_scaler(
            f"{self.channel}_scaler.pkl"
        )

        windows = self.window_generator.create_windows(train)

        tensor_dataset = TelemetryDataset(windows)

        loader = DataLoader(
            tensor_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
        )

        return loader, windows

    def train(self):

        loader, windows = self.prepare_data()

        input_size = windows.shape[2]

        model = LSTMAutoencoder(
            input_size=input_size
        ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        criterion = nn.MSELoss()

        best_loss = float("inf")

        history = []

        start_time = time.time()

        for epoch in range(Config.EPOCHS):

            model.train()

            epoch_loss = 0.0

            progress = tqdm(loader)

            progress.set_description(
                f"Epoch {epoch+1}/{Config.EPOCHS}"
            )

            for batch in progress:

                batch = batch.to(self.device)

                optimizer.zero_grad()

                output = model(batch)

                loss = criterion(output, batch)

                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()

                progress.set_postfix(
                    loss=f"{loss.item():.6f}"
                )

            epoch_loss /= len(loader)

            history.append(epoch_loss)

            print(
                f"Epoch {epoch+1:03d} | Loss : {epoch_loss:.6f}"
            )

            if epoch_loss < best_loss:

                best_loss = epoch_loss

                model_path = (
                    Config.TRAINED_MODEL_DIR /
                    f"{self.channel}_model.pth"
                )

                model.save(model_path)

                print("[OK] Best model saved.")

        elapsed = time.time() - start_time

        print("\nTraining Completed")
        print(f"Best Loss : {best_loss:.6f}")
        print(f"Time      : {elapsed:.2f} sec")

        # --------------------------------------------
        # Save History
        # --------------------------------------------

        history_path = (
            Config.METRICS_DIR /
            f"{self.channel}_history.json"
        )

        with open(history_path, "w") as f:

            json.dump(
                {
                    "channel": self.channel,
                    "loss": history,
                    "best_loss": best_loss,
                    "epochs": Config.EPOCHS,
                    "training_time": elapsed,
                },
                f,
                indent=4,
            )

        print("History Saved")

        # --------------------------------------------
        # Reconstruction Errors
        # --------------------------------------------

        model.load(
            Config.TRAINED_MODEL_DIR /
            f"{self.channel}_model.pth",
            self.device,
        )

        model.eval()

        with torch.no_grad():

            tensor = torch.tensor(
                windows,
                dtype=torch.float32,
            ).to(self.device)

            errors = model.reconstruction_error(
                tensor
            ).cpu().numpy()

        np.save(
            Config.PREDICTION_DIR /
            f"{self.channel}_train_errors.npy",
            errors,
        )

        print("Training Errors Saved")

        return model


def main():

    trainer = Trainer("A-1")

    trainer.train()


if __name__ == "__main__":

    main()