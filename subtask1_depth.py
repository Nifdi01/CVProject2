from pathlib import Path

from visualization import compare_depth_heatmaps


def build_key(path):
    parts = path.stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else path.stem


def find_pairs(left_dir, right_dir):
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    left_files = [
        p for p in left_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]
    right_files = [
        p for p in right_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    right_map = {build_key(p): p for p in right_files}
    pairs = []

    for left in sorted(left_files):
        key = build_key(left)
        right = right_map.get(key)
        if right is None:
            print(f"No matching right image for {left.name}")
            continue
        pairs.append((left, right, key))

    return pairs


def main():
    base_dir = Path(__file__).resolve().parent
    left_dir = base_dir / "images" / "left"
    right_dir = base_dir / "images" / "right"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not left_dir.exists() or not right_dir.exists():
        raise FileNotFoundError("Expected images/left and images/right folders.")

    pairs = find_pairs(left_dir, right_dir)
    if not pairs:
        raise FileNotFoundError("No matching left/right image pairs found.")

    for left_path, right_path, key in pairs:
        output_path = output_dir / f"{key}_midas_vs_stereo_heatmap.png"
        print(f"Processing {left_path.name} and {right_path.name} -> {output_path.name}")
        compare_depth_heatmaps(
            str(left_path),
            str(right_path),
            output_path=str(output_path),
            device="cpu",  # use "cpu" if no GPU
            window_size=7,
            max_disparity=256,
            aggregate_window=49,
            show=False,
        )


if __name__ == "__main__":
    main()