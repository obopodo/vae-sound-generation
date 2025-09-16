from datetime import datetime

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from soundgen.ae import Autoencoder
from soundgen.train import train
from soundgen.utils import get_device
from soundgen.vae import VAE, logger, vae_loss


def load_mnist_data(root="./data", batch_size: int = 4, return_loaders: bool = True) -> tuple[DataLoader, DataLoader]:
    train = datasets.MNIST(root=root, train=True, download=True, transform=ToTensor())
    test = datasets.MNIST(root=root, train=False, download=True, transform=ToTensor())
    if not return_loaders:
        return train, test

    train = DataLoader(train, batch_size=batch_size, shuffle=True)
    test = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train, test


if __name__ == "__main__":
    from pathlib import Path

    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005
    EPOCHS = 50
    MODEL_FILE_NAME = "vae_mnist.json"
    MODEL_CLASS = VAE  # VAE or Autoencoder

    top_folder = Path("/Users/borispodolnyi/Documents/coding_projects/vae_sound_generation/")

    train_data_loader, valid_data_loader = load_mnist_data(root=top_folder / "data/", batch_size=BATCH_SIZE)

    model = MODEL_CLASS(
        input_shape=[1, 28, 28],
        conv_filters_number=[32, 64, 64, 64],
        conv_kernel_size=[3, 3, 3, 3],
        conv_strides=[1, 2, 2, 1],
        latent_space_dim=16,
    )

    device = get_device()
    model = model.to(device)
    print("Using device:", device)

    if MODEL_CLASS == VAE:
        loss = vae_loss
    else:
        loss = nn.MSELoss()

    ########### LOAD CHECKPOINT IF NEEDED ############
    # top_folder = Path("/Users/borispodolnyi/Documents/coding_projects/vae_sound_generation/")
    # checkpoints_folder = top_folder / "models" / "20250916_123423"
    # weights_path = checkpoints_folder / "checkpoint_e050.pth"
    # params_path = checkpoints_folder / "vae_mnist.json"

    # model = MODEL_CLASS.load(weights_path, params_path).to(device)
    ##################################################

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    models_folder = top_folder / "models" / datetime.now().strftime("%Y%m%d_%H%M%S")
    models_folder.mkdir(parents=True, exist_ok=True)
    model.save_parameters(models_folder / MODEL_FILE_NAME)
    train(
        model=model,
        train_data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
        loss_fn=loss,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
        save_folder=models_folder,
    )
