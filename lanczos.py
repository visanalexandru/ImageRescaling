# This module implements various functions used to
# perform lanczos interpolation.

import numpy as np
import matplotlib.pyplot as plt
import numbers
import time


def lanczos_kernel(x, a):
    """
    Applies the lanczos kernel for the given values.

    L(x) = sinc(x)*sinc(x/a) if -a < x < a
         = 0 otherwise

    See lanczos_kernel_example() for an example.
    """

    assert type(x) == np.ndarray
    assert isinstance(a, numbers.Number)

    result = np.sinc(x) * np.sinc(x / a)
    result[x <= -a] = 0
    result[x >= a] = 0

    return result


def interpolate_lanczos(signal, at, a):
    """
    Given a one-dimensional signal with samples Si, for integer values of i,
    compute the values at arbitrary real arguments.

    Check out https://en.wikipedia.org/wiki/Lanczos_resampling
    for the interpolation formula.

    See lanczos_interpolate_example() for an example.
    """

    assert type(signal) == np.ndarray
    assert signal.ndim == 1
    assert type(at) == np.ndarray
    assert at.ndim == 1

    result = np.zeros_like(at)
    for x_index, x in enumerate(at):
        floor_x = int(np.floor(x))

        i = np.array(range(floor_x - a + 1, floor_x + a + 1))
        contributions = lanczos_kernel(x - i, a)

        # i may be out of the signal's bounds.
        # Various methods may be used in this situation.
        # We choose to clip i.
        si = signal[np.clip(i, 0, len(signal) - 1)]
        result[x_index] += np.sum(si * contributions)

    return result


def lanczos_kernel2(x, y, a):
    """
    Applies the 2d lanczos kernel for the given values.

    L(x, y) = L(x)*L(y)

    See lanczos_kernel2_example() for an example.
    """

    l_x = lanczos_kernel(x, a)
    l_y = lanczos_kernel(y, a)

    return l_x * l_y


def interpolate_lanczos2(signal, at_x, at_y, a):
    """
    Given a two-dimensional signal with samples Sij, for integer values of i and j,
    compute the values at arbitrary 2d real arguments.

    See lanczos_interpolate2_example() for an example.

    Parameters
    ----
    at_x, at_y : np.ndarray
           A 2d grid representing the x/y coordinates of the points to interpolate.

    See lanczos_interpolate2_example() for an example.
    """

    assert type(signal) == np.ndarray
    assert signal.ndim == 2

    assert type(at_x) == np.ndarray
    assert at_x.ndim == 2

    assert type(at_y) == np.ndarray
    assert at_y.ndim == 2

    assert at_x.shape == at_y.shape

    sig_height, sig_width = signal.shape
    height, width = at_x.shape
    result = np.zeros_like(at_x)

    for y in range(height):
        for x in range(width):
            # The current point to evaluate.
            x_here = at_x[y, x]
            y_here = at_y[y, x]

            floor_x = int(np.floor(x_here))
            floor_y = int(np.floor(y_here))

            # Get the coordinates around the current point.
            i, j = np.meshgrid(
                range(floor_y - a + 1, floor_y + a + 1),
                range(floor_x - a + 1, floor_x + a + 1),
                sparse=True,
            )
            contributions = lanczos_kernel2(x_here - j, y_here - i, a)

            # Get the actual signal values around the current point.
            i = np.clip(i, 0, sig_height - 1)
            j = np.clip(j, 0, sig_width - 1)
            in_signal = signal[i, j]

            result[y][x] += np.sum(contributions * in_signal)

    return result


def interpolate_lanczos2_fast(signal, at_x, at_y, a):
    """
    Same as interpolate_lanczos2(), but faster by using vector operations.
    """
    assert type(signal) == np.ndarray
    assert signal.ndim == 2

    assert type(at_x) == np.ndarray
    assert at_x.ndim == 2

    assert type(at_y) == np.ndarray
    assert at_y.ndim == 2

    assert at_x.shape == at_y.shape

    sig_height, sig_width = signal.shape
    height, width = at_x.shape
    result = np.zeros_like(at_x)

    offset_x, offset_y = np.meshgrid(
        range(-a + 1, a + 1),
        range(-a + 1, a + 1),
        sparse=True,
    )

    distances_x = np.zeros(shape=(height, width, 2 * a, 2 * a))
    distances_y = np.zeros(shape=(height, width, 2 * a, 2 * a))
    values_around = np.zeros(shape=(height, width, 2 * a, 2 * a))

    for y in range(height):
        for x in range(width):
            # The current point to evaluate.
            x_here = at_x[y, x]
            y_here = at_y[y, x]

            floor_x = int(np.floor(x_here))
            floor_y = int(np.floor(y_here))

            # Get the coordinates around the current point.
            i = offset_y + floor_y
            j = offset_x + floor_x

            # Compute the distances around the current point.
            distances_x[y, x] = x_here - j
            distances_y[y, x] = y_here - i

            # Get the actual signal values around the current point.
            i = np.clip(i, 0, sig_height - 1)
            j = np.clip(j, 0, sig_width - 1)
            values_around[y, x] = signal[i, j]

    contrib = lanczos_kernel2(distances_x, distances_y, a)
    result = np.sum(contrib * values_around, axis=(2, 3))
    return result


# Examples:


def lanczos_kernel_example():
    fig, axs = plt.subplots(4, 2)

    time = np.linspace(-20, 20, 2000)
    a_vals = [2, 4, 8, 16]

    for index, a in enumerate(a_vals):
        lanczos_samples = lanczos_kernel(time, a)

        fft = np.abs(np.fft.fftshift(np.fft.fft(lanczos_samples)))
        freq = (1 / 2) * np.fft.fftshift(np.fft.fftfreq(2000, 1 / 100))

        axs[index, 0].set_title(f"a = {a}")
        axs[index, 0].plot(time, lanczos_samples)

        axs[index, 1].set_title("Response")
        axs[index, 1].plot(freq[800:1200], fft[800:1200])
    fig.tight_layout()
    plt.show()


def lanczos_interpolate_example():
    # Generate some signal.
    domain = np.linspace(-10, 10, 1000)

    signal_f = lambda x: np.sin(x) * 3 + ((x**2) / 10) + np.cos(x) / 10
    smooth_signal = signal_f(domain)
    original_indices = np.arange(0, 1000, 1)

    # Downsample it with a factor of 100.
    downsampled_indices = np.append(np.arange(0, 1000, 100), 999)
    downsampled = smooth_signal[downsampled_indices]

    a = 4
    at = np.arange(0, 10.5, 0.5)
    upsampled = interpolate_lanczos(downsampled, at, a)

    fig, axs = plt.subplots(2, 1)
    axs[0].set_aspect(0.2, adjustable='box')
    axs[0].locator_params(axis="x", nbins=20)
    axs[0].set_title("Original")
    axs[0].plot(original_indices/100, smooth_signal, color=plt.cm.gray(0.7))
    axs[0].stem(downsampled,  linefmt='black', basefmt='gray')

    axs[1].set_aspect(0.2, adjustable='box')
    axs[1].locator_params(axis="x", nbins=20)
    axs[1].set_title("Upsampled")
    axs[1].plot(original_indices/100, smooth_signal, color=plt.cm.gray(0.7))
    axs[1].stem(at, upsampled, linefmt='black', basefmt='gray')

    fig.savefig("1D_upsampling.pdf")

    fig.tight_layout()
    plt.show()


def lanczos_kernel2_example():
    ox = np.linspace(-10, 10, 1000)
    oy = np.linspace(-10, 10, 1000)

    a = 30
    x_coords, y_coords = np.meshgrid(ox, oy)

    lanczos = lanczos_kernel2(x_coords, y_coords, a)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        x_coords, y_coords, lanczos, linewidth=0, antialiased=True, cmap=plt.cm.coolwarm
    )
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    plt.show()


def lanczos_interpolate2_example():
    signal_f = lambda x, y: 80 * np.sin(x * y)

    old_height = 100
    old_width = 100

    new_height = 120
    new_width = 300

    # First generate the signal of shape (old_height, old_width)
    oy = np.linspace(-10, 10, old_height)
    ox = np.linspace(-10, 10, old_width)
    points_x, points_y = np.meshgrid(ox, oy)
    signal = signal_f(points_x, points_y)

    # Then upscale it to (new_height, new_width)
    at_x = np.linspace(0, old_width, new_width)
    at_y = np.linspace(0, old_height, new_height)
    at_x, at_y = np.meshgrid(at_x, at_y)

    x = time.time()
    upscaled = interpolate_lanczos2_fast(signal, at_x, at_y, 2)
    print(f"Rescaling took: {time.time() - x}")

    fig, axs = plt.subplots(2, 1)

    axs[0].set_title(f"{old_width}x{old_height}")
    axs[0].imshow(signal)
    axs[1].set_title(f"{new_width}x{new_height}")
    axs[1].imshow(upscaled)
    fig.tight_layout()
    plt.show()

def DoRotation(xspan, yspan, RotRad=0):
    """Generate a meshgrid and rotate it by RotRad radians."""
    
    # Clockwise, 2D rotation matrix
    RotMatrix = np.array(
        [[np.cos(RotRad), np.sin(RotRad)], [-np.sin(RotRad), np.cos(RotRad)]]
    )
    
    x, y = np.meshgrid(xspan, yspan)
    return np.einsum("ji, mni -> jmn", RotMatrix, np.dstack([x, y]))

def meshgrid_upscale_demo():
    fig, axs = plt.subplots(1, 2)

    x = np.arange(0, 10)
    y = np.arange(0, 10)
    xx, yy = np.meshgrid(x, y)

    axs[0].set_title("Original grid")
    axs[0].set_aspect(1, adjustable='box')
    axs[0].set_xlim((-0.5,9.5))
    axs[0].locator_params(axis="x", nbins=18)
    axs[0].locator_params(axis="y", nbins=18)
    axs[0].plot((0, 9), (0,0), color=plt.cm.gray(0.7))
    axs[0].plot((9, 9), (0,9), color=plt.cm.gray(0.7))
    axs[0].plot((9, 0), (9,9), color=plt.cm.gray(0.7))
    axs[0].plot((0, 0), (9,0), color=plt.cm.gray(0.7))
    axs[0].plot(xx, yy, ".k")

    x = np.arange(0, 9.5, 0.5)
    y = np.arange(0, 9.5, 0.5)
    xx, yy = np.meshgrid(x, y)

    axs[1].set_title("Upscaled grid")
    axs[1].set_aspect(1, adjustable='box')
    axs[1].set_xlim((-0.5,9.5))
    axs[1].locator_params(axis="x", nbins=10)
    axs[1].locator_params(axis="y", nbins=10)
    axs[1].plot((0, 9), (0,0), color=plt.cm.gray(0.7))
    axs[1].plot((9, 9), (0,9), color=plt.cm.gray(0.7))
    axs[1].plot((9, 0), (9,9), color=plt.cm.gray(0.7))
    axs[1].plot((0, 0), (9,0), color=plt.cm.gray(0.7))
    axs[1].plot(xx, yy, ".k")
    plt.show()

    fig.savefig("upscale.pdf")

def meshgrid_rotate_demo():
    fig, axs = plt.subplots(1, 2)

    x = np.arange(0, 10)
    y = np.arange(0, 10)
    xx, yy = np.meshgrid(x, y)

    axs[0].set_title("Original grid")
    axs[0].set_aspect(1, adjustable='box')
    axs[0].set_xlim((-0.5,9.5))
    axs[0].locator_params(axis="x", nbins=18)
    axs[0].locator_params(axis="y", nbins=18)
    axs[0].plot((0, 9), (0,0), color=plt.cm.gray(0.7))
    axs[0].plot((9, 9), (0,9), color=plt.cm.gray(0.7))
    axs[0].plot((9, 0), (9,9), color=plt.cm.gray(0.7))
    axs[0].plot((0, 0), (9,0), color=plt.cm.gray(0.7))
    axs[0].plot(xx, yy, ".k")

    xx, yy = DoRotation(x-5, y-5, 0.3) 
    xx+=5
    yy+=5

    axs[1].set_title("Rotated grid")
    axs[1].set_aspect(1, adjustable='box')
    axs[1].set_xlim((-3,12))
    axs[1].locator_params(axis="x", nbins=10)
    axs[1].locator_params(axis="y", nbins=10)
    axs[1].plot((0, 9), (0,0), color=plt.cm.gray(0.7))
    axs[1].plot((9, 9), (0,9), color=plt.cm.gray(0.7))
    axs[1].plot((9, 0), (9,9), color=plt.cm.gray(0.7))
    axs[1].plot((0, 0), (9,0), color=plt.cm.gray(0.7))
    axs[1].plot(xx, yy, ".k")
    plt.show()

    fig.savefig("rotation.pdf")


if __name__ == "__main__":
    lanczos_kernel_example()
    lanczos_interpolate_example()
    lanczos_kernel2_example()
    lanczos_interpolate2_example()
    meshgrid_rotate_demo()
    meshgrid_upscale_demo()
