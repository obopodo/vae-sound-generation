from sympy import flatten
from torch import nn
from torchinfo import summary


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        conv_filters_number: list[int],
        conv_kernel_size: list[int],
        conv_strides: list[int],
        latent_space_dim: int,
    ):
        super().__init__()
        self.input_channels = input_channels  # num_channels
        self.conv_filters_number = conv_filters_number  # number of filters in each conv layer, eg [2, 4, 8]
        self.conv_kernel_size = conv_kernel_size  # kernel size for each conv layer, eg [3, 5, 3]
        self.conv_strides = conv_strides  # stride for each conv layer, eg [1, 2, 2]
        self.latent_space_dim = latent_space_dim

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        # self.model = nn.Sequential(self.encoder, self.decoder)
        self.model = nn.Sequential(self.encoder)

        self._num_conv_layers = len(conv_filters_number)
        self._build()

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
                in_channels = self.input_channels
            else:
                in_channels = self.conv_filters_number[i - 1]

            self.encoder.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.conv_filters_number[i],
                    kernel_size=self.conv_kernel_size[i],
                    stride=self.conv_strides[i],
                    padding=1,
                )
            )
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.BatchNorm2d(self.conv_filters_number[i]))

    def _add_bottleneck(self):
        self.encoder.append(nn.Flatten())
        self.encoder.append(nn.LazyLinear(self.latent_space_dim))

    def _build_decoder(self):
        pass

    def _build_autoencoder(self):
        pass


if __name__ == "__main__":
    model = Autoencoder(
        input_channels=1,
        conv_filters_number=[32, 64, 64, 64],
        conv_kernel_size=[3, 3, 3, 3],
        conv_strides=[1, 2, 2, 1],
        latent_space_dim=2,
    )
    summary(model, input_size=(1, 1, 28, 28))

    a = 1
