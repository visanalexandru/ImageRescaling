from rescale_image_example import rescale_rgb_image,rescale_greyscale_image
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt
import scipy.datasets as data

original_image =  data.face() # ratonul nostru preferat

# to be fast is to not concern oneself with large shapes!!
scale_factor = 2
new_height = original_image.shape[0] // scale_factor
new_width = original_image.shape[1] // scale_factor
original_image = cv2.resize(original_image, (new_width, new_height))

def mse(original, resized):
    return np.mean((original - resized) ** 2)

def mae(original, resized):
    return np.mean(np.abs(original - resized))
    
# Resizing methods dictionary, with cv2 methods, and the naive scalling
resizing_methods = {
    "Nearest": lambda img, h, w: cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST),
    "Linear": lambda img, h, w: cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR),
    "Cubic": lambda img, h, w: cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC),
    "Lanczos": lambda img, h, w: cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4),
    "Area": lambda img, h, w: cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA),
}

for kernel_size in [2,4,6,8,10]:
    resizing_methods[f"Custom lanczos kernel_size={kernel_size}"] = \
        lambda img, w, h, ks=kernel_size: rescale_rgb_image(img, w, h, ks)

def stats_rescale(new_width, new_height):
    global original_image

    

    def evaluate_metrics(original, resized):
        # Calculate metrics
        psnr = compare_psnr(original, resized)
        ssim = compare_ssim(original, resized, channel_axis=2)
        mse_val = mse(original, resized)
        mae_val = mae(original, resized)

        return {"PSNR": psnr, "SSIM": ssim, "MSE": mse_val, "MAE": mae_val}

    # Evaluating each method
    results = {}
    for method_name, resize_func in resizing_methods.items():
        print(f"Running {method_name}")
        original_shape = original_image.shape
        downsampled = resize_func(original_image, new_width, new_height)
        recovered  = resize_func(downsampled, original_shape[0], original_shape[1])
        results[method_name] = evaluate_metrics(original_image, recovered)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title('Original Image')

        plt.subplot(1, 3, 2)
        plt.imshow(downsampled)
        plt.title('Downsampled')

        plt.subplot(1, 3, 3)
        plt.imshow(recovered)
        plt.title('Recovered')

        plt.suptitle(method_name)
        plt.savefig(f'./results/up_down/{method_name}.pdf')
        plt.close()

    results_df = pd.DataFrame(results).T  

    descriptions = {
        "PSNR": "Higher is better. Ratio of maximum possible power of a signal to the power of corrupting noise.",
        "SSIM": "Higher is better. Measures the similarity between two images.",
        "MSE": "Lower is better. Average squared difference between the original and resized images.",
        "MAE": "Lower is better. Average absolute difference between the original and resized images."
    }

    print(results_df)
    print()
    for metric,desc in descriptions.items():
        print(f"{metric} : {desc}")

from pypiqe import piqe


def score(img):
    score, _, _, _ = piqe(img)
    return  score

rescale_factors = [0.25,0.5,0.75,1,1.25,1.5]

def plot_resizing_evaluation(methods, original_image):
    plt.figure(figsize=(8, 4))

    original_score = score(original_image)

    for method_name in methods:
        piqe_scores = []

        for factor in rescale_factors:
            new_height = int(original_image.shape[0] * factor)
            new_width = int(original_image.shape[1] * factor)

            print(f"compute for {method_name}, scale factor {factor}")
            resized_image = resizing_methods[method_name](original_image, new_width, new_height)
            piqe_scores.append(score(resized_image))

        plt.plot(rescale_factors, piqe_scores, marker='o', label=f'{method_name} Method')

    plt.axhline(y=original_score, color='gray', linestyle='--', label='Original Image')

    plt.xlabel('Rescale Factor')
    plt.ylabel('PIQE Score')
    plt.title('PIQE Scores for Resizing Methods')
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage:
plot_resizing_evaluation(['Nearest','Cubic', 'Linear','Lanczos','Custom lanczos kernel_size=2','Custom lanczos kernel_size=4','Custom lanczos kernel_size=8'], original_image)

exit(0)
def create_custom_image(width=20, height=30):
    image = np.zeros((height, width))

    for i in range(min(width, height)):
        image[i, i] = 255

    half_height = height // 2
    half_width = width // 2
    image[half_height, :] = 255
    image[:, half_width] = 255

    for y in range(half_height):
        for x in range(half_width):
            image[y, x] = 0

    for i in range(half_height):
        for j in range(i + 1):
            image[i, j] = 255

    return image

image = create_custom_image()
import math

imgs = {"Original" : image}

factor = 2
for method_name, resize_func in resizing_methods.items():
    new_height = int(image.shape[1] * factor)
    new_width = int(image.shape[0] * factor)
    upscaled = resize_func(image, new_width, new_height)
    imgs[method_name] = upscaled

num_images = len(imgs)
num_columns = 3
num_rows = math.ceil(num_images / num_columns)

fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 5))
axes = axes.flatten()  

for ax, (name, img) in zip(axes, imgs.items()):
    ax.imshow(img, cmap='gray')
    ax.set_title(name)
    ax.axis('off')

# Hide any unused subplots
for ax in axes[len(imgs):]:
    ax.axis('off')

plt.show()

