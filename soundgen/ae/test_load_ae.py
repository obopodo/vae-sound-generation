from torchinfo import summary

from soundgen.ae import Autoencoder

if __name__ == "__main__":
    weights_path = (
        "/Users/borispodolnyi/Documents/coding_projects/vae_sound_generation/models/checkpoint_e15_20250820_230445.pth"
    )
    params_path = "/Users/borispodolnyi/Documents/coding_projects/vae_sound_generation/models/autoencoder_params_20250820_233255.json"

    model = Autoencoder.load(weights_path, params_path)
    summary(model, input_size=[1] + list(model.input_shape))
