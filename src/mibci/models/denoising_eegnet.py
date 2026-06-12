"""Denoising-EEGNet — my own opinionated model.

Three-sentence rationale:
  1. Single-trial EEG is dominated by non-task noise, and that noise differs
     between subjects, which is exactly what wrecks subject-independent (LOSO)
     accuracy.
  2. So I prepend a small convolutional autoencoder that is pretrained to
     reconstruct each clean epoch from a noise-corrupted copy (a classic
     denoising objective), learning to suppress noise before classification.
  3. Its cleaned output feeds a standard EEGNet backbone, and the whole thing is
     then fine-tuned end-to-end — the front-end is a learned denoiser, not a
     fixed filter.

The design is justified by an ABLATION (eegnet vs denoising_eegnet on identical
CV splits), so the front-end has to earn its place rather than be assumed useful.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import DeepConfig
from ..logging_utils import get_logger
from .eegnet import EEGNet

log = get_logger()


class DenoiseFrontEnd(nn.Module):
    """Tiny conv autoencoder operating on (N, 1, C, T); returns same shape.

    Temporal-only convolutions (kernel (1,k)) so it denoises along time without
    smearing across electrodes, preserving the spatial structure EEGNet needs.
    """

    def __init__(self, latent_channels: int, kernel_length: int) -> None:
        super().__init__()
        k = kernel_length
        self.encoder = nn.Sequential(
            nn.Conv2d(1, latent_channels, (1, k), padding="same"),
            nn.ELU(),
            nn.Conv2d(latent_channels, latent_channels, (1, k), padding="same"),
            nn.ELU(),
        )
        self.decoder = nn.Conv2d(latent_channels, 1, (1, k), padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class DenoisingEEGNet(nn.Module):
    def __init__(self, n_channels: int, n_times: int, n_classes: int, cfg: DeepConfig) -> None:
        super().__init__()
        self.frontend = DenoiseFrontEnd(cfg.ae_latent_channels, cfg.kernel_length)
        self.backbone = EEGNet(n_channels, n_times, n_classes, cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.frontend(x))

    def pretrain(self, x: torch.Tensor, cfg: DeepConfig, device: torch.device) -> None:
        """Unsupervised denoising pretraining of the front-end only.

        Objective: reconstruct the clean epoch from a Gaussian-corrupted copy.
        Trains only the autoencoder weights; the backbone is untouched here and
        learns during the subsequent supervised phase.
        """
        opt = torch.optim.Adam(self.frontend.parameters(), lr=cfg.lr)
        loss_fn = nn.MSELoss()
        n = x.shape[0]
        self.frontend.train()
        for _ in range(cfg.ae_pretrain_epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, cfg.batch_size):
                idx = perm[start:start + cfg.batch_size]
                clean = x[idx]
                noisy = clean + cfg.ae_noise_std * torch.randn_like(clean)
                opt.zero_grad()
                recon = self.frontend(noisy)
                loss = loss_fn(recon, clean)
                loss.backward()
                opt.step()
        log.info("denoising front-end pretrained (%d epochs, noise_std=%.2f)",
                 cfg.ae_pretrain_epochs, cfg.ae_noise_std)


def build_denoising_eegnet(n_channels: int, n_times: int, n_classes: int,
                           cfg: DeepConfig) -> DenoisingEEGNet:
    return DenoisingEEGNet(n_channels, n_times, n_classes, cfg)
