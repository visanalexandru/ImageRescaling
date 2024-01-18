import lanczos
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib

matplotlib.use("TkAgg")


def load_array_image(path_to_image, mode="RGB"):
    if mode == "greyscale":
        return np.array(Image.open(path_to_image).convert('L'))
    return np.array(Image.open(path_to_image))


def calculate_fft(image):
    return 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft2(image))))


def rescale_greyscale_image(image, new_height, new_width, a):
    # Get the dimension of original image
    old_height, old_width = image.shape

    at_x = np.linspace(0, old_width, new_width)
    at_y = np.linspace(0, old_height, new_height)
    at_x, at_y = np.meshgrid(at_x, at_y)

    # Apply Lanczos interpolation
    upscaled_image = lanczos.interpolate_lanczos2_fast(image, at_x, at_y, a)
    upscaled_image = np.clip(upscaled_image, 0, 255)
    upscaled_image = upscaled_image.astype(np.uint8)

    return upscaled_image


def rescale_rgb_image(image, new_height, new_width, a):
    # Get number of channels for original image
    ch_number = image.shape[2]

    upscaled_channels = []
    for channel in range(ch_number):
        # Apply Lanczos interpolation to each channel
        upscaled_channel = rescale_greyscale_image(image[:, :, channel], new_height, new_width, a)
        upscaled_channel = np.clip(upscaled_channel, 0, 255)
        upscaled_channels.append(upscaled_channel)

    # Stack the upscaled channels
    upscaled_image = (np.stack(upscaled_channels, axis=-1)).astype(np.uint8)

    return upscaled_image


if __name__ == "__main__":
    input_image_path = 'bunny.png'
    image = load_array_image(input_image_path)

    a = 2
    new_width = 800
    new_height = 300

    # Rgb example
    rgb_rescaled = rescale_rgb_image(image, new_height, new_width, a)

    fig, axs = plt.subplots(2, 1)
    axs[0].set_title(f"{image.shape[0]}x{image.shape[1]}")
    axs[0].imshow(image, cmap=plt.get_cmap('gray'))

    axs[1].set_title(f"{new_height}x{new_width}")
    axs[1].imshow(rgb_rescaled, cmap=plt.get_cmap('gray'))

    fig.tight_layout()
    plt.show()

    # Greyscale example
    image = load_array_image(input_image_path, mode="greyscale")
    fft_original = calculate_fft(image)

    upscaled = rescale_greyscale_image(image, new_height, new_width, a)
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
