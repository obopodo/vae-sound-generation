import json
import logging

import numpy as np
import torch
from torch import nn
from torchinfo import summary

from soundgen.utils import calculate_conv2d_output_shape, get_device

MSE_LOSS_WEIGHT = 100
WARMUP_EPOCHS = 20

logger = logging.getLogger("vae_logger")
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("vae_loss.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

class Encoder(nn.Module):
    def __init__(
        self,
        input_shape: int,
        conv_filters_number: list[int],
        conv_kernel_size: list[int],
        conv_strides: list[int],
        latent_space_dim: int,
        padding: int = 1,
    ):
        super().__init__()
        self.input_shape = input_shape  # [num_channels, height, width]
        self.num_channels = input_shape[0]
        self.conv_filters_number = conv_filters_number  # number of filters in each conv layer, eg [2, 4, 8]
        self.conv_kernel_size = conv_kernel_size  # kernel size for each conv layer, eg [3, 5, 3]
        self.conv_strides = conv_strides  # stride for each conv layer, eg [1, 2, 2]
        self.latent_space_dim = latent_space_dim
        self.padding = padding

        self._num_conv_layers = len(conv_filters_number)
        self.shape_before_bottleneck = self._get_shape_before_bottleneck()
        # size of the flattened layer after convolutions
        self.linear_layer_size = np.prod(self.shape_before_bottleneck)

        self.mu = nn.Linear(self.linear_layer_size, self.latent_space_dim)
        self.log_var = nn.Linear(self.linear_layer_size, self.latent_space_dim)
        self.conv_layers = self._init_convolutions()

    def _get_shape_before_bottleneck(self):
        """Calculate the size of the flattened layer after convolutions.

        The same will be used for the layer output size after the bottleneck.
        """
        size = self.input_shape[1:]  # the number of input channels doesn't matter
        for kernel_size, stride in zip(self.conv_kernel_size, self.conv_strides):
            size = calculate_conv2d_output_shape(size, kernel_size, padding=self.padding, stride=stride)
        return self.conv_filters_number[-1], size[0], size[1]  # (last_conv_channels, height, width)

    def _init_convolutions(self) -> nn.Sequential:
        conv_layers = nn.Sequential()
        for i in range(self._num_conv_layers):
            if i == 0:
                in_channels = self.num_channels
            else:
                in_channels = self.conv_filters_number[i - 1]

            conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.conv_filters_number[i],
                    kernel_size=self.conv_kernel_size[i],
                    stride=self.conv_strides[i],
                    padding=self.padding,
                )
            )
            conv_layers.append(nn.BatchNorm2d(self.conv_filters_number[i]))
            conv_layers.append(nn.ReLU())

        conv_layers.append(nn.Flatten())
        return conv_layers

    def forward(self, x):
        x = self.conv_layers(x)
        means = self.mu(x)
        log_vars = self.log_var(x)
        return means, log_vars


class Decoder(nn.Module):
    def __init__(
        self,
        latent_space_dim: int,
        shape_before_bottleneck: tuple,
        linear_layer_size: int,
        conv_filters_number: list[int],
        conv_kernel_size: list[int],
        conv_strides: list[int],
        num_channels: int,
        padding: int = 1,
    ):
        super().__init__()

        self.conv_filters_number = conv_filters_number  # number of filters in each conv layer, eg [2, 4, 8]
        self.conv_kernel_size = conv_kernel_size  # kernel size for each conv layer, eg [3, 5, 3]
        self.conv_strides = conv_strides  # stride for each conv layer, eg [1, 2, 2]
        self._num_conv_layers = len(conv_filters_number)
        self.latent_space_dim = latent_space_dim
        self.num_channels = num_channels
        self.padding = padding

        self.dense_layer = nn.Linear(latent_space_dim, linear_layer_size)
        self.reshape_layer = nn.Unflatten(1, shape_before_bottleneck)
        self.conv_transpose_layers = self._init_conv_transpose_layers()
        self.sigmoid = nn.Sigmoid()

    def _init_conv_transpose_layers(self) -> nn.Sequential:
        """Loop through the all conv layers in reverse order except the first one."""
        conv_transpose_layers = nn.Sequential()
        for i in reversed(range(1, self._num_conv_layers)):
            conv_transpose_layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.conv_filters_number[i],  # ?
                    out_channels=self.conv_filters_number[i - 1],  # ?
                    kernel_size=self.conv_kernel_size[i],
                    stride=self.conv_strides[i],
                    padding=self.padding,
                    output_padding=(1 if self.conv_strides[i] > 1 else 0),
                )
                # output padding is used to ensure the output shape matches the input shape
                # IDK how to calculate the exact value, 1 just worked for stride equal to 1 and 2
            )
            conv_transpose_layers.append(nn.ReLU())
            conv_transpose_layers.append(nn.BatchNorm2d(self.conv_filters_number[i - 1]))

        # Add the last conv transpose layer to get back to the original number of channels
        conv_transpose_layers.append(
            nn.ConvTranspose2d(
                in_channels=self.conv_filters_number[0],
                out_channels=self.num_channels,
                kernel_size=self.conv_kernel_size[0],
                stride=self.conv_strides[0],
                padding=self.padding,
            )
        )
        return conv_transpose_layers

    def forward(self, x):
        x = self.dense_layer(x)
        x = self.reshape_layer(x)
        x = self.conv_transpose_layers(x)
        x = self.sigmoid(x)
        return x


class VAE(nn.Module):

    def __init__(
        self,
        input_shape: int,
        conv_filters_number: list[int],
        conv_kernel_size: list[int],
        conv_strides: list[int],
        latent_space_dim: int,
        padding: int = 1,
    ):
        super().__init__()
        self._mu = None
        self._log_var = None
        self.input_shape = input_shape  # [num_channels, height, width]
        self.num_channels = input_shape[0]
        self.conv_filters_number = conv_filters_number  # number of filters in each conv layer, eg [2, 4, 8]
        self.conv_kernel_size = conv_kernel_size  # kernel size for each conv layer, eg [3, 5, 3]
        self.conv_strides = conv_strides  # stride for each conv layer, eg [1, 2, 2]
        self.latent_space_dim = latent_space_dim
        self.padding = padding

        self.encoder = Encoder(
            input_shape=input_shape,
            conv_filters_number=conv_filters_number,
            conv_kernel_size=conv_kernel_size,
            conv_strides=conv_strides,
            latent_space_dim=latent_space_dim,
            padding=padding,
        )
        self.decoder = Decoder(
            latent_space_dim=latent_space_dim,
            shape_before_bottleneck=self.encoder.shape_before_bottleneck,
            linear_layer_size=self.encoder.linear_layer_size,
            conv_filters_number=conv_filters_number,
            conv_kernel_size=conv_kernel_size,
            conv_strides=conv_strides,
            num_channels=self.num_channels,
            padding=padding,
        )

    def forward(self, x):
        self._mu, self._log_var = self.encoder(x)
        x = self.reparameterize(self._mu, self._log_var)
        x = self.decoder(x)
        return x

    def reparameterize(self, mu, log_var):
        epsilon = torch.randn_like(log_var)  # .to(DEVICE)
        z = mu + torch.exp(log_var / 2) * epsilon  # reparameterization trick
        return z

    def save(self, weights_path: str, params_path: str):
        self.save_parameters(params_path)
        torch.save(self.state_dict(), weights_path)

    def save_weights(self, weights_path: str):
        torch.save(self.state_dict(), weights_path)

    def save_parameters(self, params_path: str):
        params = {
            "input_shape": self.input_shape,
            "conv_filters_number": self.conv_filters_number,
            "conv_kernel_size": self.conv_kernel_size,
            "conv_strides": self.conv_strides,
            "latent_space_dim": self.latent_space_dim,
            "padding": self.padding,
        }
        with open(params_path, "w") as f:
            json.dump(params, f, indent=4)

    @classmethod
    def load(cls, weights_path: str, params_path: str):
        with open(params_path, "r") as f:
            params = json.load(f)
        model = cls(**params)
        model.load_state_dict(torch.load(weights_path))
        return model


def vae_loss(preds, X, mu, log_var, epoch: int) -> torch.Tensor:
    """Calculate VAE loss as a sum of reconstruction loss (MSE) and KL divergence."""
    reconstruction_loss = nn.functional.mse_loss(preds, X, reduction="mean")
    kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    kl_beta = max(0, min(1.0, (epoch - 1) / WARMUP_EPOCHS))
    logger.info(f"Epoch {epoch} | MSE: {reconstruction_loss.item()} | KLD: {kl_divergence.item()} | Beta: {kl_beta}")
    return MSE_LOSS_WEIGHT * reconstruction_loss + kl_beta * kl_divergence


if __name__ == "__main__":
    model = VAE(
        input_shape=[1, 28, 28],
        conv_filters_number=[32, 64, 64, 64],
        conv_kernel_size=[3, 3, 3, 3],
        conv_strides=[1, 2, 2, 1],
        latent_space_dim=16,
    )
    device = get_device()
    model = model.to(device)
    summary(model, input_size=(1, 1, 28, 28))
