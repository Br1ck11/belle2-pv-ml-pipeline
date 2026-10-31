"""Convolutional variational autoencoder (VAE) for multivariate PV windows.

Input/output: float32 (B, C, T) windows, as stored in dataset.h5.

Encoder: ``len(hidden_channels)`` Conv1d layers. kernel_size, stride,
padding and activation are each either ONE value shared by all layers or a
list with one value per layer. Layer i maps widths[i] -> widths[i + 1]
channels and time length L -> floor((L + 2 p - k) / s) + 1, then applies
activation i ("linear" = none). padding "same" means k // 2 (odd k), which
keeps L for stride 1 and gives ceil(L / s) otherwise. The final feature map
is flattened and mapped linearly to the posterior mean ``mu`` and
log-variance ``logvar`` of a diagonal Gaussian q(z | x).

Decoder: the mirror image. A linear map from z to the final feature-map
shape followed by the activation of the LAST encoder layer, then one
ConvTranspose1d per encoder layer in reverse order; the one that rebuilds
the input of encoder layer i is followed by activation i - 1, and the output
layer (back to C channels) is linear because z-scored values are unbounded.
``output_padding`` is chosen per layer so every intermediate length, and the
reconstruction's T, match the encoder exactly.

Output layer (output_kernel_size = k_out > 0, default 7): the last transposed
convolution keeps the first hidden width (hidden_channels[0]) and is followed
by activation 0; a final stride-1 Conv1d(hidden_channels[0] -> C, k_out,
padding k_out // 2) without activation produces the reconstruction. A
stride-2 transposed convolution computes even and odd output samples with
different kernel taps, which creates a period-2 ("checkerboard") pattern;
the stride-1 convolution uses all of its taps at every sample and can smooth
it away. output_kernel_size = 0 restores the earlier decoder, whose last
transposed convolution maps directly to C channels.

Objective (per window, then averaged over the batch):
    reconstruction = 0.5 * sum_{c,t} (x_hat - x)^2
        = Gaussian negative log-likelihood with unit variance in the stored
          (normalized) units, up to a constant (= 0.5 * C * T * MSE)
    kl = KL(q(z|x) || N(0, I)) = -0.5 * sum_j (1 + logvar - mu^2 - exp(logvar))
    loss = reconstruction + beta * kl         (beta = 1: the standard ELBO)
A smaller beta is equivalent to assuming a smaller observation noise.

In train mode z is sampled with the reparameterisation trick. In eval mode
z = mu, so validation losses and later anomaly scores are deterministic.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from ..base import Batch, ModelSpec, TrainableModel


ACTIVATIONS = {"linear": nn.Identity, "relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}
Activation = Literal["linear", "relu", "silu", "gelu"]
Padding = int | Literal["same"]


def _per_layer(value, n_layers: int, name: str) -> tuple:
    """One shared value -> n_layers copies; a list must have one entry per layer."""
    if isinstance(value, tuple):
        if len(value) != n_layers:
            raise ValueError(f"model.{name} has {len(value)} entries but hidden_channels has {n_layers} layers")
        return value
    return (value,) * n_layers


@dataclass(frozen=True)
class CNNVAEHyperparameters:
    hidden_channels: tuple[int, ...]                     # encoder widths, e.g. [8, 12, 16]
    kernel_size: int | tuple[int, ...]                   # one value or one per layer
    stride: int | tuple[int, ...]                        # temporal downsampling per layer
    latent_dim: int                                      # dimension of z
    beta: float                                          # KL weight
    activation: Activation | tuple[Activation, ...] = "silu"
    padding: Padding | tuple[Padding, ...] = "same"      # "same" = kernel_size // 2
    output_kernel_size: int = 7                          # final stride-1 Conv1d (odd); 0 = none

    def __post_init__(self):
        n_layers = len(self.hidden_channels)
        if any(width <= 0 for width in self.hidden_channels):
            raise ValueError("model.hidden_channels must contain positive integers")
        for kernel, stride, padding in zip(self.kernels, self.strides,
                                           _per_layer(self.padding, n_layers, "padding")):
            if kernel <= 0:
                raise ValueError("model.kernel_size must contain positive integers")
            if padding == "same" and kernel % 2 == 0:
                raise ValueError("padding \"same\" needs an odd kernel_size")
            if not 1 <= stride <= kernel:
                # A stride above the kernel size would skip input samples entirely.
                raise ValueError("model.stride must satisfy 1 <= stride <= kernel_size in every layer")
            if padding != "same" and not 0 <= padding <= kernel // 2:
                raise ValueError("model.padding must be \"same\" or an integer in [0, kernel_size // 2]")
        _per_layer(self.activation, n_layers, "activation")
        if self.latent_dim <= 0:
            raise ValueError("model.latent_dim must be positive")
        if self.beta < 0:
            raise ValueError("model.beta must be nonnegative")
        if self.output_kernel_size < 0 or (self.output_kernel_size and self.output_kernel_size % 2 == 0):
            raise ValueError("model.output_kernel_size must be 0 (no output layer) or a positive odd integer")

    @property
    def kernels(self) -> tuple[int, ...]:
        return _per_layer(self.kernel_size, len(self.hidden_channels), "kernel_size")

    @property
    def strides(self) -> tuple[int, ...]:
        return _per_layer(self.stride, len(self.hidden_channels), "stride")

    @property
    def paddings(self) -> tuple[int, ...]:
        """Integer padding per layer ("same" resolved to kernel_size // 2)."""
        paddings = _per_layer(self.padding, len(self.hidden_channels), "padding")
        return tuple(k // 2 if p == "same" else p for k, p in zip(self.kernels, paddings))

    @property
    def activations(self) -> tuple[str, ...]:
        return _per_layer(self.activation, len(self.hidden_channels), "activation")


def encoder_lengths(window_size: int, kernels, strides, paddings) -> list[int]:
    """Time length before the first and after every encoder layer:
    Conv1d gives floor((L + 2p - k) / s) + 1."""
    lengths = [window_size]
    for kernel, stride, padding in zip(kernels, strides, paddings):
        length = (lengths[-1] + 2 * padding - kernel) // stride + 1
        if length <= 0:
            raise ValueError(f"Window size T = {window_size} is too short for the encoder (lengths {lengths})")
        lengths.append(length)
    return lengths


class CNNVAE(TrainableModel):
    def __init__(self, hparams: CNNVAEHyperparameters, *, n_channels: int, window_size: int) -> None:
        super().__init__()
        if n_channels <= 0 or window_size <= 0:
            raise ValueError("n_channels and window_size must be positive")
        self.hparams = hparams
        self.n_channels = n_channels
        self.window_size = window_size
        kernels, strides, paddings = hparams.kernels, hparams.strides, hparams.paddings
        activations = [ACTIVATIONS[name] for name in hparams.activations]

        # widths[i] -> widths[i + 1] and lengths[i] -> lengths[i + 1] in encoder layer i.
        self.widths = (n_channels, *hparams.hidden_channels)
        self.lengths = encoder_lengths(window_size, kernels, strides, paddings)
        self.feature_shape = (self.widths[-1], self.lengths[-1])
        flat = self.widths[-1] * self.lengths[-1]

        encoder = []
        for i, (kernel, stride, padding) in enumerate(zip(kernels, strides, paddings)):
            encoder += [nn.Conv1d(self.widths[i], self.widths[i + 1], kernel, stride=stride, padding=padding),
                        activations[i]()]
        self.encoder = nn.Sequential(*encoder)
        self.to_mu = nn.Linear(flat, hparams.latent_dim)
        self.to_logvar = nn.Linear(flat, hparams.latent_dim)

        self.from_latent = nn.Linear(hparams.latent_dim, flat)
        self.latent_activation = activations[-1]()   # mirrors the last encoder layer
        decoder = []
        k_out = hparams.output_kernel_size
        for i in reversed(range(len(hparams.hidden_channels))):
            # ConvTranspose1d gives (L_in - 1) * s - 2p + k + output_padding;
            # output_padding (0 <= op < s) restores exactly L = lengths[i].
            kernel, stride, padding = kernels[i], strides[i], paddings[i]
            output_padding = self.lengths[i] - ((self.lengths[i + 1] - 1) * stride - 2 * padding + kernel)
            if not 0 <= output_padding < stride:
                raise ValueError(f"Decoder layer {i} cannot restore length {self.lengths[i]} "
                                 f"(kernel {kernel}, stride {stride}, padding {padding})")
            # With an output layer, the last transposed conv keeps hidden width widths[1].
            width_out = self.widths[i] if (i > 0 or not k_out) else self.widths[1]
            decoder.append(nn.ConvTranspose1d(
                self.widths[i + 1], width_out, kernel,
                stride=stride, padding=padding, output_padding=output_padding,
            ))
            if i > 0:
                decoder.append(activations[i - 1]())
            elif k_out:
                decoder.append(activations[0]())
        if k_out:
            # Final stride-1 convolution back to C channels; linear (z-scored values are unbounded).
            decoder.append(nn.Conv1d(self.widths[1], n_channels, k_out, stride=1, padding=k_out // 2))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, C, T) -> posterior mean and log-variance, each (B, latent_dim)."""
        features = self.encoder(values).flatten(start_dim=1)
        return self.to_mu(features), self.to_logvar(features)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, latent_dim) -> reconstruction (B, C, T)."""
        features = self.latent_activation(self.from_latent(z))
        return self.decoder(features.view(-1, *self.feature_shape))

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(values)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return self.decode(z), mu, logvar

    is_variational = True

    def evaluation_output(self, batch: Batch, *, latent_mode: str | None = None) -> dict[str, torch.Tensor]:
        """Reconstruction for testing: decode z = mu ("z_mean", deterministic)
        or one reparameterised sample z = mu + sigma * eps ("sampling_z")."""
        if latent_mode not in ("z_mean", "sampling_z"):
            raise ValueError("CNNVAE needs latent_mode 'z_mean' or 'sampling_z'")
        values = batch["values"]
        if values.ndim != 3 or tuple(values.shape[1:]) != (self.n_channels, self.window_size):
            raise ValueError(f"CNNVAE was built for (C, T) = ({self.n_channels}, {self.window_size}) and cannot "
                             f"process windows of shape {tuple(values.shape[1:])} (its linear layers fix T)")
        mu, logvar = self.encode(values)
        z = mu if latent_mode == "z_mean" else mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return {"output": self.decode(z), "target": values}

    def compute_losses(self, batch: Batch) -> dict[str, torch.Tensor]:
        values = batch["values"]
        if values.ndim != 3 or tuple(values.shape[1:]) != (self.n_channels, self.window_size):
            raise ValueError(f"Expected (B, {self.n_channels}, {self.window_size}) windows, got {tuple(values.shape)}")
        reconstruction, mu, logvar = self(values)
        squared_error = (reconstruction - values).square()
        reconstruction_loss = 0.5 * squared_error.sum(dim=(1, 2)).mean()
        kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum(dim=1).mean()
        return {
            "loss": reconstruction_loss + self.hparams.beta * kl,
            "reconstruction": reconstruction_loss,
            "kl": kl,
            "mse": squared_error.mean(),   # per value; comparable with the plain autoencoder
        }


def build(hparams: CNNVAEHyperparameters, *, n_channels: int, window_size: int) -> CNNVAE:
    return CNNVAE(hparams, n_channels=n_channels, window_size=window_size)


# Checkpoints written before output_kernel_size existed have no output layer.
SPEC = ModelSpec(name="cnn_vae", hyperparameters=CNNVAEHyperparameters, build=build,
                 checkpoint_defaults={"output_kernel_size": 0})
