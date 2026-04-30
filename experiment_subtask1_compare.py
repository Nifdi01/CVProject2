import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import stereo_matching


DEFAULT_CONFIGS = [
    {
        "name": "w5_d48_a7",
        "window_size": 5,
        "max_disparity": 48,
        "aggregate_window": 7,
    },
    {
        "name": "w7_d64_a9",
        "window_size": 7,
        "max_disparity": 64,
        "aggregate_window": 9,
    },
    {
        "name": "w7_d96_a9",
        "window_size": 7,
        "max_disparity": 96,
        "aggregate_window": 9,
    },
    {
        "name": "w7_d64_a11",
        "window_size": 7,
        "max_disparity": 64,
        "aggregate_window": 11,
    },
    {
        "name": "w5_d64_a9",
        "window_size": 5,
        "max_disparity": 64,
        "aggregate_window": 9,
    },
]


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


def normalize_map(values, eps=1e-6):
    v_min = float(np.nanmin(values))
    v_max = float(np.nanmax(values))
    if v_max - v_min < eps:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - v_min) / (v_max - v_min)).astype(np.float32)


def safe_corrcoef(a, b):
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_metrics(disparity, midas_depth):
    disp_norm = normalize_map(disparity)
    midas_norm = normalize_map(midas_depth)
    diff = disp_norm - midas_norm
    abs_diff = np.abs(diff)

    flat_disp = disp_norm.reshape(-1)
    flat_midas = midas_norm.reshape(-1)
    valid = np.isfinite(flat_disp) & np.isfinite(flat_midas)

    metrics = {
        "mae": float(np.mean(abs_diff)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "p50_abs_diff": float(np.percentile(abs_diff, 50)),
        "p90_abs_diff": float(np.percentile(abs_diff, 90)),
        "corr": safe_corrcoef(flat_disp[valid], flat_midas[valid]),
        "disp_min": float(np.min(disparity)),
        "disp_max": float(np.max(disparity)),
        "midas_min": float(np.min(midas_depth)),
        "midas_max": float(np.max(midas_depth)),
    }

    return metrics


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def load_midas(device):
    start = time.perf_counter()
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
    midas.to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform
    sync_if_cuda(device)
    elapsed = time.perf_counter() - start
    return midas, transform, elapsed


def midas_depth_map_from_bgr(image_bgr, midas, transform, device):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction.cpu().numpy()


def run_comparison(
    left_path,
    right_path,
    midas,
    transform,
    device,
    window_size,
    max_disparity,
    aggregate_window,
):
    left_bgr = cv2.imread(str(left_path))
    right_bgr = cv2.imread(str(right_path))
    if left_bgr is None or right_bgr is None:
        raise FileNotFoundError("Could not read left or right image.")

    sync_if_cuda(device)
    t0 = time.perf_counter()
    midas_depth = midas_depth_map_from_bgr(left_bgr, midas, transform, device)
    sync_if_cuda(device)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    disparity = stereo_matching.compute_disparity_from_images(
        left_bgr,
        right_bgr,
        window_size=window_size,
        max_disparity=max_disparity,
        aggregate_window=aggregate_window,
    )
    t3 = time.perf_counter()

    t4 = time.perf_counter()
    metrics = compute_metrics(disparity, midas_depth)
    t5 = time.perf_counter()

    return {
        "midas_depth": midas_depth,
        "disparity": disparity,
        "metrics": metrics,
        "time_midas_s": t1 - t0,
        "time_stereo_s": t3 - t2,
        "time_compare_s": t5 - t4,
        "time_total_s": (t1 - t0) + (t3 - t2) + (t5 - t4),
    }


def mean(values):
    return float(np.mean(values)) if values else float("nan")


def load_configs(path):
    if not path:
        return DEFAULT_CONFIGS

    with open(path, "r") as handle:
        data = json.load(handle)

    if isinstance(data, dict) and "configs" in data:
        data = data["configs"]

    if not isinstance(data, list):
        raise ValueError("Config file must be a list or contain a 'configs' list.")

    for idx, cfg in enumerate(data, start=1):
        if "name" not in cfg:
            cfg["name"] = f"cfg_{idx}"

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Quantitatively compare MiDaS vs stereo depth on subtask1 images."
    )
    parser.add_argument("--left-dir", default="images/left", help="Left image folder")
    parser.add_argument("--right-dir", default="images/right", help="Right image folder")
    parser.add_argument("--output-dir", default="outputs", help="Output folder")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--config",
        default="",
        help="JSON file with list of parameter configs",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Use a single parameter set from CLI arguments",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=7,
        help="Census window size (used with --single)",
    )
    parser.add_argument(
        "--max-disparity",
        type=int,
        default=64,
        help="Max disparity (used with --single)",
    )
    parser.add_argument(
        "--aggregate-window",
        type=int,
        default=9,
        help="Aggregation window (used with --single)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of pairs")
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one warmup pair and record it in CSV",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    left_dir = base_dir / args.left_dir
    right_dir = base_dir / args.right_dir
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not left_dir.exists() or not right_dir.exists():
        raise FileNotFoundError("Expected images/left and images/right folders.")

    pairs = find_pairs(left_dir, right_dir)
    if not pairs:
        raise FileNotFoundError("No matching left/right image pairs found.")

    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        device = torch.device("cpu")

    midas, transform, midas_load_s = load_midas(device)

    if args.single:
        configs = [
            {
                "name": "single",
                "window_size": args.window_size,
                "max_disparity": args.max_disparity,
                "aggregate_window": args.aggregate_window,
            }
        ]
    else:
        configs = load_configs(args.config)

    csv_path = output_dir / "subtask1_compare_metrics.csv"
    rows = []

    if args.warmup and pairs:
        left_path, right_path, key = pairs[0]
        warmup_cfg = configs[0]
        print(f"Warmup on {left_path.name} and {right_path.name}")
        result = run_comparison(
            left_path,
            right_path,
            midas,
            transform,
            device,
            warmup_cfg["window_size"],
            warmup_cfg["max_disparity"],
            warmup_cfg["aggregate_window"],
        )
        rows.append(
            {
                "key": key,
                "left_image": left_path.name,
                "right_image": right_path.name,
                "config": warmup_cfg["name"],
                "window_size": warmup_cfg["window_size"],
                "max_disparity": warmup_cfg["max_disparity"],
                "aggregate_window": warmup_cfg["aggregate_window"],
                "time_midas_s": result["time_midas_s"],
                "time_stereo_s": result["time_stereo_s"],
                "time_compare_s": result["time_compare_s"],
                "time_total_s": result["time_total_s"],
                "is_warmup": 1,
                **result["metrics"],
            }
        )
        pairs = pairs[1:]

    for cfg in configs:
        for left_path, right_path, key in pairs:
            print(
                "Processing"
                f" {left_path.name} and {right_path.name}"
                f" with {cfg['name']}"
            )
            result = run_comparison(
                left_path,
                right_path,
                midas,
                transform,
                device,
                cfg["window_size"],
                cfg["max_disparity"],
                cfg["aggregate_window"],
            )
            rows.append(
                {
                    "key": key,
                    "left_image": left_path.name,
                    "right_image": right_path.name,
                    "config": cfg["name"],
                    "window_size": cfg["window_size"],
                    "max_disparity": cfg["max_disparity"],
                    "aggregate_window": cfg["aggregate_window"],
                    "time_midas_s": result["time_midas_s"],
                    "time_stereo_s": result["time_stereo_s"],
                    "time_compare_s": result["time_compare_s"],
                    "time_total_s": result["time_total_s"],
                    "is_warmup": 0,
                    **result["metrics"],
                }
            )

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    data_rows = [r for r in rows if r.get("is_warmup") == 0]
    summary_rows = []
    by_config = {}
    for row in data_rows:
        by_config.setdefault(row["config"], []).append(row)

    for cfg in configs:
        cfg_name = cfg["name"]
        items = by_config.get(cfg_name, [])
        if not items:
            continue
        summary_rows.append(
            {
                "config": cfg_name,
                "window_size": cfg["window_size"],
                "max_disparity": cfg["max_disparity"],
                "aggregate_window": cfg["aggregate_window"],
                "images": len(items),
                "avg_mae": mean([r["mae"] for r in items]),
                "avg_rmse": mean([r["rmse"] for r in items]),
                "avg_p50_abs_diff": mean([r["p50_abs_diff"] for r in items]),
                "avg_p90_abs_diff": mean([r["p90_abs_diff"] for r in items]),
                "avg_corr": mean([r["corr"] for r in items]),
                "avg_time_midas_s": mean([r["time_midas_s"] for r in items]),
                "avg_time_stereo_s": mean([r["time_stereo_s"] for r in items]),
                "avg_time_compare_s": mean([r["time_compare_s"] for r in items]),
                "avg_time_total_s": mean([r["time_total_s"] for r in items]),
            }
        )

    summary_csv_path = output_dir / "subtask1_compare_summary.csv"
    summary_fields = list(summary_rows[0].keys()) if summary_rows else []
    with open(summary_csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary = {
        "device": str(device),
        "images": len(data_rows),
        "midas_model_load_s": midas_load_s,
        "warmup_enabled": bool(args.warmup),
        "csv_metrics": str(csv_path),
        "csv_summary": str(summary_csv_path),
        "configs": {
            row["config"]: row for row in summary_rows
        },
    }

    summary_path = output_dir / "subtask1_compare_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved metrics to {csv_path}")
    print(f"Saved summary table to {summary_csv_path}")
    print(f"Saved summary JSON to {summary_path}")


if __name__ == "__main__":
    main()
