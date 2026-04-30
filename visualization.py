import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

import midas_utils
import stereo_matching


def normalize_map(values, eps=1e-6):
    """
    Normalize a map to [0, 1] using min-max scaling.

    Args:
        values: 2D array
        eps: Small value to avoid division by zero

    Returns:
        Normalized float32 map
    """
    v_min = float(np.nanmin(values))
    v_max = float(np.nanmax(values))
    if v_max - v_min < eps:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - v_min) / (v_max - v_min)).astype(np.float32)


def compare_depth_heatmaps(
    left_image_path,
    right_image_path,
    output_path="./outputs/midas_vs_stereo_heatmap.png",
    device="cpu",
    window_size=7,
    max_disparity=64,
    aggregate_window=9,
    show=True,
):
    """
    Compare MiDaS and stereo depth with side-by-side heatmaps.

    Args:
        left_image_path: Path to left image
        right_image_path: Path to right image
        output_path: Output figure path
        device: torch device for MiDaS
        window_size: Census window size (odd)
        max_disparity: Maximum disparity range
        aggregate_window: Cost aggregation window size
        show: Whether to display the figure

    Returns:
        Dict with raw disparity, MiDaS depth, and normalized difference
    """
    left_bgr = cv2.imread(left_image_path)
    right_bgr = cv2.imread(right_image_path)
    if left_bgr is None or right_bgr is None:
        raise FileNotFoundError("Could not read left or right image.")

    disparity = stereo_matching.compute_disparity_from_images(
        left_bgr,
        right_bgr,
        window_size=window_size,
        max_disparity=max_disparity,
        aggregate_window=aggregate_window,
    )
    midas_depth = midas_utils.midas_depth_map(left_image_path, device)

    disparity_norm = normalize_map(disparity)
    midas_norm = normalize_map(midas_depth)
    difference = np.abs(disparity_norm - midas_norm)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    axes[0].imshow(left_rgb)
    axes[0].set_title("Left image")
    axes[0].axis("off")

    disp_plot = axes[1].imshow(disparity_norm, cmap="inferno")
    axes[1].set_title("Stereo disparity")
    axes[1].axis("off")
    fig.colorbar(disp_plot, ax=axes[1], fraction=0.046, pad=0.04)

    midas_plot = axes[2].imshow(midas_norm, cmap="inferno")
    axes[2].set_title("MiDaS depth")
    axes[2].axis("off")
    fig.colorbar(midas_plot, ax=axes[2], fraction=0.046, pad=0.04)

    diff_plot = axes[3].imshow(difference, cmap="magma")
    axes[3].set_title("Absolute difference")
    axes[3].axis("off")
    fig.colorbar(diff_plot, ax=axes[3], fraction=0.046, pad=0.04)

    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    return {
        "disparity": disparity,
        "midas_depth": midas_depth,
        "difference": difference,
    }


def visualize_stereo_comparison(left_img, results):
    """
    Visualize input image and multiple disparity maps side by side.

    Args:
        left_img: Input left image
        results: List of dicts with keys: disparity, window_size, max_disparity, aggregate_window
    """
    count = len(results)
    fig, axes = plt.subplots(1, count + 1, figsize=(4 * (count + 1), 4))

    # Input image with color range
    img_min = float(left_img.min())
    img_max = float(left_img.max())
    input_plot = axes[0].imshow(left_img, cmap='gray', vmin=img_min, vmax=img_max)
    axes[0].set_title(f"Input Image\nRange: [{img_min:.0f}, {img_max:.0f}]")
    axes[0].axis('off')
    fig.colorbar(input_plot, ax=axes[0])

    # Disparity maps
    for idx, result in enumerate(results, start=1):
        disparity = result["disparity"]
        window_size = result["window_size"]
        max_disparity = result["max_disparity"]
        aggregate_window = result["aggregate_window"]

        disp_plot = axes[idx].imshow(disparity, cmap='jet')
        axes[idx].set_title(
            f"Census: {window_size}\nMax disp: {max_disparity} | Agg win: {aggregate_window}"
        )
        axes[idx].axis('off')
        fig.colorbar(disp_plot, ax=axes[idx])

    plt.tight_layout()
    os.makedirs('./outputs', exist_ok=True)
    plt.savefig('./outputs/census_transform_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def save_comparison_figure(
    left_img,
    disparities,
    window_size,
    max_disparities,
    aggregate_window,
    output_dir,
):
    fig, axes = plt.subplots(1, len(max_disparities) + 1, figsize=(4 * (len(max_disparities) + 1), 4))

    img_min = float(left_img.min())
    img_max = float(left_img.max())
    input_plot = axes[0].imshow(left_img, cmap="gray", vmin=img_min, vmax=img_max)
    axes[0].set_title(f"Input\nRange: [{img_min:.0f}, {img_max:.0f}]")
    axes[0].axis("off")
    fig.colorbar(input_plot, ax=axes[0])

    for idx, (disparity, max_disparity) in enumerate(zip(disparities, max_disparities), start=1):
        disp_plot = axes[idx].imshow(disparity, cmap="jet")
        axes[idx].set_title(
            f"Census: {window_size} | Agg: {aggregate_window}\nMax disp: {max_disparity}"
        )
        axes[idx].axis("off")
        fig.colorbar(disp_plot, ax=axes[idx])

    fig.tight_layout()
    filename = f"test_w{window_size}_a{aggregate_window}.png"
    fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)