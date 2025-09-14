from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from torchinfo import summary

from soundgen.ae import Autoencoder
from soundgen.scripts.train_script import load_mnist_data
from soundgen.vae import VAE


def tensor_to_image(image: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a numpy image supported by matplotlib.

    squeeze(0) removes batch dimension
    permute(1, 2, 0) makes channel dimension last
    """
    return image.squeeze(0).permute(1, 2, 0).detach().numpy()


def plot_one_sample(real_image: torch.Tensor, reconstructed: torch.Tensor, label: int, ax: Axes):
    real_image = tensor_to_image(real_image)
    reconstructed = tensor_to_image(reconstructed)
    ax[0].imshow(real_image, cmap="gray")
    ax[0].set_title(f"{label}: Original Image")
    ax[1].imshow(reconstructed, cmap="gray")
    ax[1].set_title(f"Reconstructed Image")


if __name__ == "__main__":
    N_SAMPLES = 10
    RANDOM_SEED = 73
    MODEL_CLASS = VAE  # VAE or Autoencoder

    top_folder = Path("/Users/borispodolnyi/Documents/coding_projects/vae_sound_generation/models")
    weights_path = top_folder / "checkpoint_e17_20250914_155332.pth"
    params_path = top_folder / "vae_mnist_20250914_154615.json"

    model = MODEL_CLASS.load(weights_path, params_path)
    # summary(model, input_size=[1] + list(model.input_shape))

    _, test_dataset = load_mnist_data(root="./data", return_loaders=False)

    np.random.seed(RANDOM_SEED)
    _, ax = plt.subplots(2, N_SAMPLES, figsize=(6 * N_SAMPLES, 4))
    for plot_index, sample_index in enumerate(np.random.randint(len(test_dataset), size=N_SAMPLES)):
        image, label = test_dataset[sample_index]
        image = image.unsqueeze(0)
        reconstructed_image = model(image)
        plot_one_sample(image, reconstructed_image, label, ax[:, plot_index])
    plt.tight_layout()
    plt.show()
