import json

import numpy as np
import torch
from torch import nn
from torchinfo import summary

from soundgen.utils import calculate_conv2d_output_shape


class Autoencoder(nn.Module):

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

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        # self.model = nn.Sequential(self.encoder)
        self.model = nn.Sequential(self.encoder, self.decoder)

        self._num_conv_layers = len(conv_filters_number)
        self._shape_before_bottleneck = self._get_shape_before_bottleneck()
        self._linear_layer_size = np.prod(
            self._shape_before_bottleneck
        )  # size of the flattened layer after convolutions
        self._build()

    def _get_shape_before_bottleneck(self):
        """Calculate the size of the flattened layer after convolutions.

        The same will be used for the layer output size after the bottleneck.
        """
        size = self.input_shape[1:]  # the number of input channels doesn't matter
        for kernel_size, stride in zip(self.conv_kernel_size, self.conv_strides):
            size = calculate_conv2d_output_shape(size, kernel_size, padding=self.padding, stride=stride)
        return self.conv_filters_number[-1], size[0], size[1]  # (last_conv_channels, height, width)

    def forward(self, x):
        x = self.model(x)
        return x

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_encoder(self):
        self._add_convolutions()
        self._add_bottleneck()

    def _add_convolutions(self):
        for i in range(self._num_conv_layers):
            if i == 0:
                in_channels = self.num_channels
            else:
                in_channels = self.conv_filters_number[i - 1]

            self.encoder.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.conv_filters_number[i],
                    kernel_size=self.conv_kernel_size[i],
                    stride=self.conv_strides[i],
                    padding=self.padding,
                )
            )
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.BatchNorm2d(self.conv_filters_number[i]))

    def _add_bottleneck(self):
        self.encoder.append(nn.Flatten())
        # self.encoder.append(nn.LazyLinear(self.latent_space_dim))
        self.encoder.append(nn.Linear(self._linear_layer_size, self.latent_space_dim))

    def _build_decoder(self):
        # self.decoder.append(nn.Linear(self.latent_space_dim, ))
        self._add_dense_layer()
        self._add_reshape_layer()
        self._add_conv_transpose_layers()
        self._add_decoder_output()

    def _add_dense_layer(self):
        self.decoder.append(nn.Linear(self.latent_space_dim, self._linear_layer_size))

    def _add_reshape_layer(self):
        self.decoder.append(nn.Unflatten(1, self._shape_before_bottleneck))

    def _add_conv_transpose_layers(self):
        """Loop through the all conv layers in reverse order except the first one."""
        for i in reversed(range(1, self._num_conv_layers)):
            self.decoder.append(
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
            self.decoder.append(nn.ReLU())
            self.decoder.append(nn.BatchNorm2d(self.conv_filters_number[i - 1]))

    def _add_decoder_output(self):
        self.decoder.append(
            nn.ConvTranspose2d(
                in_channels=self.conv_filters_number[0],
                out_channels=self.num_channels,
                kernel_size=self.conv_kernel_size[0],
                stride=self.conv_strides[0],
                padding=self.padding,
            )
        )
        self.decoder.append(nn.Sigmoid())  # Use Sigmoid for output layer to normalize the output

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
            json.dump(params, f)

    @classmethod
    def load(cls, weights_path: str, params_path: str):
        with open(params_path, "r") as f:
            params = json.load(f)
        model = cls(**params)
        model.load_state_dict(torch.load(weights_path))
        return model


if __name__ == "__main__":
    model = Autoencoder(
        input_shape=[1, 28, 28],
        conv_filters_number=[32, 64, 64, 64],
        conv_kernel_size=[3, 3, 3, 3],
        conv_strides=[1, 2, 2, 1],
        latent_space_dim=2,
    )
    summary(model, input_size=(1, 1, 28, 28))

    a = 1
