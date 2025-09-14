from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Slider

from soundgen.ae import Autoencoder
from soundgen.vae import VAE


def xy_to_tensor(x: float, y: float) -> torch.Tensor:
    """Convert x, y coordinates to a tensor."""
    return torch.tensor([[x, y]]).float()


if __name__ == "__main__":
    top_folder = Path("/Users/borispodolnyi/Documents/coding_projects/vae_sound_generation/models")
    weights_path = top_folder / "checkpoint_e17_20250914_155332.pth"
    params_path = top_folder / "vae_mnist_20250914_154615.json"
    MODEL_CLASS = VAE  # VAE or Autoencoder

    model = MODEL_CLASS.load(weights_path, params_path)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    def latent_to_image(model: VAE | Autoencoder, x: float, y: float) -> np.ndarray:
        latent_vector = xy_to_tensor(x, y)
        reconstructed_image = model.decoder(latent_vector)
        reconstructed_image = reconstructed_image.squeeze(0).permute(1, 2, 0).detach().numpy()
        img = ax.imshow(reconstructed_image, cmap="gray")
        fig.canvas.draw()

    fig.subplots_adjust(left=0.25, bottom=0.25)

    x_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    x_slider = Slider(
        ax=x_ax,
        label="1st latent coordinate",
        valmin=-50.0,
        valmax=50.0,
        valinit=0.0,
    )

    y_ax = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    y_slider = Slider(
        ax=y_ax,
        label="2nd latent coordinate",
        valmin=-50.0,
        valmax=50.0,
        valinit=0.0,
        orientation="vertical",
    )

    x_slider.on_changed(lambda val: latent_to_image(model, val, y_slider.val))
    y_slider.on_changed(lambda val: latent_to_image(model, x_slider.val, val))
    plt.show()
