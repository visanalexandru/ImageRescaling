import lanczos
from rescale_image_example import load_array_image, calculate_fft
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")


# Taken from: https://stackoverflow.com/questions/29708840/rotate-meshgrid-with-numpy
def DoRotation(xspan, yspan, RotRad=0):
    """Generate a meshgrid and rotate it by RotRad radians."""

    # Clockwise, 2D rotation matrix
    RotMatrix = np.array(
        [[np.cos(RotRad), np.sin(RotRad)], [-np.sin(RotRad), np.cos(RotRad)]]
    )

    x, y = np.meshgrid(xspan, yspan)
    return np.einsum("ji, mni -> jmn", RotMatrix, np.dstack([x, y]))


def rescale_and_rotate_greyscale_image(image, new_height, new_width, angle):
    old_height = image.shape[0]
    old_width = image.shape[1]

    at_x = np.arange(0, old_width, old_width / new_width)
    at_y = np.arange(0, old_height, old_height / new_height)

    at_x, at_y = DoRotation(at_x - old_width / 2, at_y - old_height / 2, angle)
    at_x = at_x + old_width / 2
    at_y = at_y + old_height / 2

    upscaled_image = lanczos.interpolate_lanczos2_fast(image, at_x, at_y, 2)

    return upscaled_image


def rescale_and_rotate_rgb_image(image, new_height, new_width, angle):
    ch_number = image.shape[2]

    upscaled_channels = []
    for channel in range(ch_number):
        # Apply Lanczos interpolation to each channel
        upscaled_channel = rescale_and_rotate_greyscale_image(image[:, :, channel], new_height, new_width, angle)
        upscaled_channel = np.clip(upscaled_channel, 0, 255)
        upscaled_channels.append(upscaled_channel)

    # Stack the upscaled channels
    upscaled_image = (np.stack(upscaled_channels, axis=-1)).astype(np.uint8)

    return upscaled_image


if __name__ == "__main__":
    input_image_path = 'bunny.png'
    image = load_array_image(input_image_path)

    new_height = 400
    new_width = 400
    angle = 0.8

    upscaled = rescale_and_rotate_rgb_image(image, new_width, new_height, angle)

    fig, axs = plt.subplots(2, 1)

    axs[0].set_title(f"{image.shape[0]}x{image.shape[1]}")
    axs[0].imshow(image)
    axs[1].set_title(f"{new_width}x{new_height}")
    axs[1].imshow(upscaled)
    fig.tight_layout()
    plt.show()

    image = load_array_image(input_image_path, mode="greyscale")
    fft_original = calculate_fft(image)

    upscaled = rescale_and_rotate_greyscale_image(image, new_height, new_width, angle)
    fft_upscale = calculate_fft(upscaled)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].set_title(f"{image.shape[0]}x{image.shape[1]}")
    axs[0, 0].imshow(image, cmap=plt.get_cmap('gray'))

    axs[0, 1].set_title(f"FFT 2D")
    axs[0, 1].imshow(fft_original)

    axs[1, 0].set_title(f"{new_height}x{new_width}")
    axs[1, 0].imshow(upscaled, cmap=plt.get_cmap('gray'))

    axs[1, 1].set_title(f"FFT 2D")
    axs[1, 1].imshow(fft_upscale)

    fig.tight_layout()
    plt.show()
