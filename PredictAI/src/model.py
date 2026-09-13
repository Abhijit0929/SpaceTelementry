"""
model.py

LSTM Autoencoder for Spacecraft Telemetry Anomaly Detection
"""

from typing import Tuple

import torch
import torch.nn as nn

from src.config import Config


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder

    Input
        (batch, seq_len, features)

    Output
        (batch, seq_len, features)
    """

    def __init__(self, input_size: int):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = Config.HIDDEN_SIZE
        self.latent_size = Config.LATENT_SIZE
        self.num_layers = Config.NUM_LAYERS

        # -------------------------------------------------
        # Encoder
        # -------------------------------------------------

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=Config.DROPOUT,
        )

        self.latent = nn.Linear(
            self.hidden_size,
            self.latent_size,
        )

        # -------------------------------------------------
        # Decoder preparation
        # -------------------------------------------------

        self.hidden_projection = nn.Linear(
            self.latent_size,
            self.hidden_size,
        )

        # -------------------------------------------------
        # Decoder
        # -------------------------------------------------

        self.decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=Config.DROPOUT,
        )

        self.output_layer = nn.Linear(
            self.hidden_size,
            input_size,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch_size = x.size(0)
        seq_len = x.size(1)

        # ----------------------------------------
        # Encoder
        # ----------------------------------------

        _, (hidden, cell) = self.encoder(x)

        latent = self.latent(hidden[-1])

        # ----------------------------------------
        # Decoder Initial Hidden State
        # ----------------------------------------

        decoder_hidden = self.hidden_projection(latent)

        decoder_hidden = decoder_hidden.unsqueeze(0)

        decoder_hidden = decoder_hidden.repeat(
            self.num_layers,
            1,
            1,
        )

        decoder_cell = torch.zeros_like(decoder_hidden)

        # ----------------------------------------
        # Decoder Input
        # ----------------------------------------

        decoder_input = torch.zeros(
            batch_size,
            seq_len,
            self.input_size,
            device=x.device,
        )

        decoded, _ = self.decoder(
            decoder_input,
            (decoder_hidden, decoder_cell),
        )

        reconstruction = self.output_layer(decoded)

        return reconstruction

    @torch.no_grad()
    def reconstruct(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        self.eval()

        return self.forward(x)

    @torch.no_grad()
    def reconstruction_error(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        reconstruction = self.forward(x)

        error = torch.mean(

            (x - reconstruction) ** 2,

            dim=(1, 2),

        )

        return error

    def save(
        self,
        path,
    ):

        torch.save(
            self.state_dict(),
            path,
        )

    def load(
        self,
        path,
        device,
    ):

        self.load_state_dict(

            torch.load(
                path,
                map_location=device,
            )

        )

        self.eval()

    @property
    def num_parameters(self):

        return sum(

            p.numel()

            for p in self.parameters()

            if p.requires_grad

        )


if __name__ == "__main__":

    model = LSTMAutoencoder(
        input_size=25,
    )

    x = torch.randn(
        8,
        Config.WINDOW_SIZE,
        25,
    )

    y = model(x)

    print("=" * 60)
    print("LSTM Autoencoder")
    print("=" * 60)

    print("Input :", x.shape)
    print("Output:", y.shape)
    print("Trainable Parameters :", model.num_parameters)