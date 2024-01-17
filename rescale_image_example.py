from skimage.color import rgb2gray

import lanczos
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib

matplotlib.use("TkAgg")


def calculate_fft(image):
    return 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft2(image))))


def lanczos_rgb_image_example(image, new_height, new_width):

    # Get the dimension and channels of original images
    old_height = image.shape[0]
    old_width = image.shape[1]
    ch_number = image.shape[2]

    upscaled_channels = []
    for channel in range(ch_number):
        at_x = np.linspace(0, old_width, new_width)
        at_y = np.linspace(0, old_height, new_height)
        at_x, at_y = np.meshgrid(at_x, at_y)

        # Apply Lanczos interpolation to each channel
        upscaled_channel = lanczos.interpolate_lanczos2_fast(
            original_array[:, :, channel], at_x, at_y, 2
        )
        upscaled_channel = np.clip(upscaled_channel, 0, 255)
        upscaled_channels.append(upscaled_channel)

    # Stack the upscaled channels
    upscaled = np.stack(upscaled_channels, axis=-1)
    upscaled = upscaled.astype(np.uint8)

    return upscaled


def lanczos_grayscale_image_example(image, new_height, new_width):
    # Get the dimension and channels of original images
    old_height = image.shape[0]
    old_width = image.shape[1]

    at_x = np.linspace(0, old_width, new_width)
    at_y = np.linspace(0, old_height, new_height)
    at_x, at_y = np.meshgrid(at_x, at_y)

    # Apply Lanczos interpolation to each channel
    upscaled = lanczos.interpolate_lanczos2(original_array, at_x, at_y, 2)
    upscaled = np.clip(upscaled, 0, 255)
    upscaled = upscaled.astype(np.uint8)

    return upscaled


if __name__ == "__main__":
    # lanczos_rgb_image_example(original_array, new_height, new_width)

    # Load the image
    input_image_path = 'bunny.png'
    original_image = Image.open(input_image_path).convert('L')
    # Convert the image to a NumPy array
    original_array = np.array(original_image)
    original_height = original_array.shape[0]
    original_width = original_array.shape[1]
    fft_original = calculate_fft(original_array)
    new_width = 60
    new_height = 30
    upscaled = lanczos_grayscale_image_example(original_array, new_height, new_width)
    fft_upscale = calculate_fft(upscaled)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].set_title(f"{original_height}x{original_width}")
    axs[0, 0].imshow(original_array, cmap=plt.get_cmap('gray'))
    axs[0, 0].imshow(original_array, cmap=plt.get_cmap('gray'))
    axs[0, 1].imshow(fft_original)
    axs[1, 0].set_title(f"{new_height}x{new_width}")
    axs[1, 0].imshow(upscaled, cmap=plt.get_cmap('gray'))
    axs[1, 1].imshow(fft_upscale)
    fig.tight_layout()
    plt.show()
