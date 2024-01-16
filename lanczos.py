# This module implements various functions used to
# perform lanczos interpolation.

import numpy as np
import matplotlib.pyplot as plt
import numbers


def lanczos_kernel(x, a):
    """
    Applies the lanczos kernel for the given values.

    L(x) = sinc(x)*sinc(x/a) if -a < x < a
         = 0 otherwise

    See lanczos_kernel_example() for an example.
    """

    assert type(x) == np.ndarray
    assert isinstance(a, numbers.Number)

    result = np.zeros_like(x)
    mask = np.logical_and(x > -a, x < a)
    in_range = x[mask]
    result[mask] = np.sinc(in_range) * np.sinc(in_range / a)

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


# Examples:


def lanczos_kernel_example():
    fig, axs = plt.subplots(4, 1)

    time = np.linspace(-20, 20, 2000)
    a_vals = [4, 8, 10, 20]

    for index, a in enumerate(a_vals):
        lanczos_samples = lanczos_kernel(time, a)
        axs[index].set_title(f"a = {a}")
        axs[index].plot(time, lanczos_samples)
    fig.tight_layout()
    plt.show()


def lanczos_interpolate_example():
    # Generate some signal.
    signal_f = lambda x: np.sin(x) * 3 + ((x**2) / 10) + np.cos(x) / 10
    smooth_signal = signal_f(np.linspace(-10, 10, 1000))

    downsample_factor = 100
    upsample_factor = 100
    a = 4

    # Downsample it.
    downsampled_indices = np.arange(0, 1000, downsample_factor)
    downsampled = smooth_signal[downsampled_indices]

    # Upsample it.
    at = np.linspace(0, len(downsampled), len(downsampled) * upsample_factor)
    upsampled = interpolate_lanczos(downsampled, at, a)

    fig, axs = plt.subplots(3, 1)
    axs[0].set_title("Original")
    axs[0].plot(smooth_signal)

    axs[1].set_title(f"{downsample_factor}x downsampled")
    axs[1].plot(downsampled_indices, downsampled)

    axs[2].set_title(f"{upsample_factor}x upsampled")
    axs[2].plot(upsampled)

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


if __name__ == "__main__":
    lanczos_kernel_example()
    lanczos_interpolate_example()
    lanczos_kernel2_example()
