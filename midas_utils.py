import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def save_heatmap(depth, out_path, cmap="inferno"):
    """
    Save a depth heatmap with a colorbar.

    Args:
        depth: 2D depth map
        out_path: Output image path
        cmap: Matplotlib colormap name
    """
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    heat = ax.imshow(depth, cmap=cmap)
    ax.axis("off")
    fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def midas_depth_map(image_path, device):
    """
    Compute MiDaS relative depth map from a single image.

    Args:
        image_path: Path to input image
        device: torch device ("cpu" or "cuda")

    Returns:
        Depth map as a numpy array
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Could not read image for MiDaS.")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
    midas.to(device).eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform

    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    return depth


def midas_depth(image_path, out_path, device):
    depth = midas_depth_map(image_path, device)
    save_heatmap(depth, out_path)
    return depth
