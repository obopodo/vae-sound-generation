from datetime import datetime

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from soundgen.ae import Autoencoder
from soundgen.train import train
from soundgen.utils import get_device


def load_mnist_data(root="./data", batch_size: int = 4, return_loaders: bool = True) -> tuple[DataLoader, DataLoader]:
    train = datasets.MNIST(root=root, train=True, download=True, transform=ToTensor())
    test = datasets.MNIST(root=root, train=False, download=True, transform=ToTensor())
    if not return_loaders:
        return train, test

    train = DataLoader(train, batch_size=batch_size, shuffle=True)
    test = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train, test


if __name__ == "__main__":
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005
    EPOCHS = 20

    train_data_loader, valid_data_loader = load_mnist_data(batch_size=BATCH_SIZE)

    model = Autoencoder(
        input_shape=[1, 28, 28],
        conv_filters_number=[32, 64, 64, 64],
        conv_kernel_size=[3, 3, 3, 3],
        conv_strides=[1, 2, 2, 1],
        latent_space_dim=2,
    )

    device = get_device()
    print("Using device:", device)
    model = model.to(device)

    loss = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    model.save_parameters(f"./models/autoencoder_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    train(
        model=model,
        train_data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
        loss_fn=loss,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
        save_checkpoint=True,
    )
