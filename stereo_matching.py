import numpy as np
import cv2
from numba import njit, prange


def census_transform(gray, window_size=7):
    """
    Compute the Census transform for a grayscale image.

    Args:
        gray: 2D grayscale image
        window_size: Odd window size (<= 7 to fit in 64 bits)

    Returns:
        2D array of uint64 Census descriptors
    """
    if gray.ndim != 2:
        raise ValueError("Census transform expects a 2D grayscale image.")
    if window_size % 2 == 0:
        raise ValueError("Census window_size must be odd.")

    num_bits = window_size * window_size - 1
    if num_bits > 64:
        raise ValueError("Census window_size too large for uint64 descriptor.")

    h, w = gray.shape
    pad = window_size // 2
    padded = np.pad(gray, pad, mode="edge")
    census = np.zeros((h, w), dtype=np.uint64)

    for dy in range(window_size):
        for dx in range(window_size):
            if dy == pad and dx == pad:
                continue
            neighbor = padded[dy:dy + h, dx:dx + w]
            census = (census << 1) | (neighbor < gray).astype(np.uint64)

    return census


@njit(cache=True)
def hamming_distance_scalar(a, b):
    """
    Compute Hamming distance between two values (Numba optimized).
    
    Args:
        a: First value
        b: Second value
        
    Returns:
        Hamming distance as integer
    """
    xor = a ^ b
    count = 0
    while xor:
        count += xor & 1
        xor >>= 1
    return count


@njit(parallel=True, cache=True)
def compute_cost_volume(census_left, census_right, h, w, max_disparity, max_cost):
    """
    Compute cost volume using parallel Hamming distance computation.
    
    Args:
        census_left: Census transform of left image
        census_right: Census transform of right image
        h: Image height
        w: Image width
        max_disparity: Maximum disparity range
        max_cost: Maximum cost value for invalid disparities
        
    Returns:
        Cost volume array
    """
    cost_volume = np.zeros((h, w, max_disparity), dtype=np.float32)
    
    for d in prange(max_disparity):
        for y in range(h):
            for x in range(w):
                if x - d >= 0:
                    cost_volume[y, x, d] = hamming_distance_scalar(
                        census_left[y, x], 
                        census_right[y, x - d]
                    )
                else:
                    cost_volume[y, x, d] = max_cost
    
    return cost_volume


def compute_disparity(
    census_left,
    census_right,
    max_disparity=64,
    aggregate_window=9,
    max_cost=None,
):
    """
    Compute disparity map with cost aggregation using parallelism.
    
    Args:
        census_left: Census transform of left image
        census_right: Census transform of right image
        max_disparity: Maximum disparity range
        aggregate_window: Window size for cost aggregation
        max_cost: Maximum cost value for invalid disparities
        
    Returns:
        Tuple of (disparity map, raw cost volume, aggregated cost volume)
    """
    h, w = census_left.shape
    if max_cost is None:
        max_cost = 48  # Max cost for 7x7 window (48 bits)
    
    # Build cost volume with parallelism
    print("Building cost volume (parallel)...")
    cost_volume = compute_cost_volume(census_left, census_right, h, w, max_disparity, max_cost)
    
    # Cost aggregation using box filter
    print("Aggregating costs...")
    aggregated_cost = np.zeros_like(cost_volume)
    
    for d in range(max_disparity):
        # Simple box filter aggregation
        aggregated_cost[:, :, d] = cv2.boxFilter(
            cost_volume[:, :, d], 
            -1, 
            (aggregate_window, aggregate_window)
        )
    
    # Winner-takes-all: select disparity with minimum cost
    print("Computing disparity map...")
    disparity = np.argmin(aggregated_cost, axis=2).astype(np.float32)
    
    return disparity, cost_volume, aggregated_cost


def compute_disparity_from_images(
    left_img,
    right_img,
    window_size=7,
    max_disparity=64,
    aggregate_window=9,
):
    """
    Convenience wrapper to compute disparity from image arrays.

    Args:
        left_img: Left image (BGR or grayscale)
        right_img: Right image (BGR or grayscale)
        window_size: Census window size (odd)
        max_disparity: Maximum disparity range
        aggregate_window: Window size for cost aggregation

    Returns:
        Disparity map
    """
    left_gray = left_img
    right_gray = right_img

    if left_img.ndim == 3:
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    if right_img.ndim == 3:
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    census_left = census_transform(left_gray, window_size=window_size)
    census_right = census_transform(right_gray, window_size=window_size)
    max_cost = window_size * window_size - 1

    disparity, _, _ = compute_disparity(
        census_left,
        census_right,
        max_disparity=max_disparity,
        aggregate_window=aggregate_window,
        max_cost=max_cost,
    )

    return disparity