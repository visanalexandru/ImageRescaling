import lanczos

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib

matplotlib.use("TkAgg")


def lanczos_rgb_image_example():
    # Load the image
    input_image_path = "bunny.png"
    original_image = Image.open(input_image_path)

    # Convert the image to a NumPy array
    original_array = np.array(original_image)

    # Get the dimension and channels of original images
    old_height = original_array.shape[0]
    old_width = original_array.shape[1]
    ch_number = original_array.shape[2]

    # Define the scale factor for rescaling
    scale_factor = 0.3

    # Calculate the new size after rescaling
    new_height = int(old_height * scale_factor)
    new_width = int(old_width * scale_factor)

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

    fig, axs = plt.subplots(2, 1)

    axs[0].set_title(f"{old_width}x{old_height}")
    axs[0].imshow(original_array)
    axs[1].set_title(f"{new_width}x{new_height}")
    axs[1].imshow(upscaled)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    lanczos_rgb_image_example()
