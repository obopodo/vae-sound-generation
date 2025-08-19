import torch


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():  # for Apple Silicon Macs
        return "mps"
    else:
        return "cpu"


def calculate_conv1d_output_shape(input_length: int, kernel_size: int, *, padding: int = 0, stride: int = 1) -> int:
    """Conv Output Size = (Input Size - Kernel Size + 2 * Padding) / Stride + 1"""
    return (input_length - kernel_size + 2 * padding) // stride + 1


def calculate_conv2d_output_shape(
    input_shape: tuple[int, int],
    kernel_size: int,
    *,
    padding: int = 0,
    stride: int = 1,
) -> tuple[int, int]:
    """
    Calculate 2D convolution output shape.

    Conv Output Size = (Input Size - Kernel Size + 2 * Padding) / Stride + 1
    Example: (64, 44) -> (16, 11)
    """
    height, width = input_shape
    output_height = (height - kernel_size + 2 * padding) // stride + 1
    output_width = (width - kernel_size + 2 * padding) // stride + 1
    return output_height, output_width


if __name__ == "__main__":
    # (64, 44) -> (16, 11)
    x = (64, 44)
    x = calculate_conv2d_output_shape(x, 3, padding=1, stride=1)
    x = calculate_conv2d_output_shape(x, 3, padding=1, stride=2)
    x = calculate_conv2d_output_shape(x, 3, padding=1, stride=2)
    print(x)
