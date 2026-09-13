"""
predict.py

Run inference using the trained LSTM Autoencoder and compute
reconstruction errors on the NASA SMAP/MSL test dataset.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from src.config import Config
from src.dataset import NASADataset, TelemetryDataset
from src.preprocessing import DataPreprocessor, WindowGenerator
from src.model import LSTMAutoencoder


def main(channel: str = "A-1"):
    print("=" * 60)
    print("SPACECRAFT INTELLIGENCE")
    print("Prediction / Inference")
    print("=" * 60)
    print(f"Channel : {channel}")
    print(f"Device  : {Config.DEVICE}")

    # Load Dataset
    dataset = NASADataset()
    _, test_data, _ = dataset.load_channel(channel)

    # Preprocessing
    preprocessor = DataPreprocessor()
    # Load the scaler fitted during training
    preprocessor.load_scaler(f"{channel}_scaler.pkl")
    
    test_scaled = preprocessor.transform(test_data)

    # Generate sliding windows
    window_gen = WindowGenerator()
    windows = window_gen.create_windows(test_scaled)
    tensor_dataset = TelemetryDataset(windows)
    
    dataloader = TorchDataLoader(
        tensor_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
    )

    # Load Model
    input_size = windows.shape[2]
    model_path = Config.TRAINED_MODEL_DIR / f"{channel}_model.pth"
    
    model = LSTMAutoencoder(input_size=input_size).to(Config.DEVICE)
    model.load(model_path, Config.DEVICE)

    # Inference (reconstruction error calculation)
    print("\nRunning inference...")
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(Config.DEVICE)
            error = model.reconstruction_error(batch)
            reconstruction_errors.extend(error.cpu().numpy())

    reconstruction_errors = np.array(reconstruction_errors)

    # Results
    print("\n" + "=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)
    print(f"Total Windows : {len(reconstruction_errors)}")
    print(f"Mean Error    : {reconstruction_errors.mean():.8f}")
    print(f"Max Error     : {reconstruction_errors.max():.8f}")
    print(f"Min Error     : {reconstruction_errors.min():.8f}")

    save_path = Config.PREDICTION_DIR / f"{channel}_test_errors.npy"
    np.save(save_path, reconstruction_errors)

    print(f"\n[OK] Reconstruction errors saved at: {save_path}")


if __name__ == "__main__":
    import sys
    target_channel = sys.argv[1] if len(sys.argv) > 1 else "A-1"
    main(target_channel)
