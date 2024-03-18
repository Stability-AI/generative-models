import os

import matplotlib.pyplot as plt
import numpy as np


def generate_dynamic_cycle_xy_values(
    length=21,
    init_elev=0,
    num_components=84,
    frequency_range=(1, 5),
    amplitude_range=(0.5, 10),
    step_range=(0, 2),
):
    # Y values generation
    y_sequence = np.ones(length) * init_elev
    for _ in range(num_components):
        # Choose a frequency that will complete whole cycles in the sequence
        frequency = np.random.randint(*frequency_range) * (2 * np.pi / length)
        amplitude = np.random.uniform(*amplitude_range)
        phase_shift = np.random.choice([0, np.pi])  # np.random.uniform(0, 2 * np.pi)
        angles = (
            np.linspace(0, frequency * length, length, endpoint=False) + phase_shift
        )
        y_sequence += np.sin(angles) * amplitude
    # X values generation
    # Generate length - 1 steps since the last step is back to start
    steps = np.random.uniform(*step_range, length - 1)
    total_step_sum = np.sum(steps)
    # Calculate the scale factor to scale total steps to just under 360
    scale_factor = (
        360 - ((360 / length) * np.random.uniform(*step_range))
    ) / total_step_sum
    # Apply the scale factor and generate the sequence of X values
    x_values = np.cumsum(steps * scale_factor)
    # Ensure the sequence starts at 0 and add the final step to complete the loop
    x_values = np.insert(x_values, 0, 0)
    return x_values, y_sequence


def smooth_data(data, window_size):
    # Extend data at both ends by wrapping around to create a continuous loop
    pad_size = window_size
    padded_data = np.concatenate((data[-pad_size:], data, data[:pad_size]))

    # Apply smoothing
    kernel = np.ones(window_size) / window_size
    smoothed_data = np.convolve(padded_data, kernel, mode="same")

    # Extract the smoothed data corresponding to the original sequence
    # Adjust the indices to account for the larger padding
    start_index = pad_size
    end_index = -pad_size if pad_size != 0 else None
    smoothed_original_data = smoothed_data[start_index:end_index]
    return smoothed_original_data


# Function to generate and process the data
def gen_dynamic_loop(length=21, elev_deg=0):
    while True:
        # Generate the combined X and Y values using the new function
        azim_values, elev_values = generate_dynamic_cycle_xy_values(
            length=84, init_elev=elev_deg
        )
        # Smooth the Y values directly
        smoothed_elev_values = smooth_data(elev_values, 5)
        max_magnitude = np.max(np.abs(smoothed_elev_values))
        if max_magnitude < 90:
            break
    subsample = 84 // length
    azim_rad = np.deg2rad(azim_values[::subsample])
    elev_rad = np.deg2rad(smoothed_elev_values[::subsample])
    # Make cond frame the last one
    return np.roll(azim_rad, -1), np.roll(elev_rad, -1)


def plot_3D(azim, polar, save_path, dynamic=True):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    elev = np.deg2rad(90) - polar
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    cm = plt.get_cmap("Greys")
    col_line = [cm(i) for i in np.linspace(0.3, 1, len(azim) + 1)]
    cm = plt.get_cmap("cool")
    col = [cm(float(i) / (len(azim))) for i in np.arange(len(azim))]
    xs = np.cos(elev) * np.cos(azim)
    ys = np.cos(elev) * np.sin(azim)
    zs = np.sin(elev)
    ax.scatter(xs[0], ys[0], zs[0], s=100, color=col[0])
    xs_d, ys_d, zs_d = (xs[1:] - xs[:-1]), (ys[1:] - ys[:-1]), (zs[1:] - zs[:-1])
    for i in range(len(xs) - 1):
        if dynamic:
            ax.quiver(
                xs[i], ys[i], zs[i], xs_d[i], ys_d[i], zs_d[i], lw=2, color=col_line[i]
            )
        else:
            ax.plot(xs[i : i + 2], ys[i : i + 2], zs[i : i + 2], lw=2, c=col_line[i])
        ax.scatter(xs[i + 1], ys[i + 1], zs[i + 1], s=100, color=col[i + 1])
    ax.scatter(xs[:1], ys[:1], zs[:1], s=120, facecolors="none", edgecolors="k")
    ax.scatter(xs[-1:], ys[-1:], zs[-1:], s=120, facecolors="none", edgecolors="k")
    ax.view_init(elev=30, azim=-20, roll=0)
    plt.savefig(save_path, bbox_inches="tight")
    plt.clf()
    plt.close()
