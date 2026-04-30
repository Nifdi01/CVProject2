import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO


DEFAULT_CONFIGS = [
    {
        "name": "base",
        "score_thresh": 0.7,
        "near_m": 1.0,
        "far_m": 50.0,
        "min_valid": 0.1,
        "max_valid": 200.0,
        "min_pixels": 25,
    },
    {
        "name": "low_thresh",
        "score_thresh": 0.5,
        "near_m": 1.0,
        "far_m": 50.0,
        "min_valid": 0.1,
        "max_valid": 200.0,
        "min_pixels": 25,
    },
    {
        "name": "high_thresh",
        "score_thresh": 0.85,
        "near_m": 1.0,
        "far_m": 50.0,
        "min_valid": 0.1,
        "max_valid": 200.0,
        "min_pixels": 25,
    },
    {
        "name": "tight_range",
        "score_thresh": 0.7,
        "near_m": 0.5,
        "far_m": 30.0,
        "min_valid": 0.1,
        "max_valid": 120.0,
        "min_pixels": 25,
    },
]


def list_images(directory):
    extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def normalize_map(values, eps=1e-6):
    v_min = float(np.nanmin(values))
    v_max = float(np.nanmax(values))
    if v_max - v_min < eps:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - v_min) / (v_max - v_min)).astype(np.float32)


def clamp_box(box, width, height):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(1, min(x2, width))
    y2 = max(1, min(y2, height))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def estimate_distance_from_depth(depth_map, box, min_valid, max_valid, min_pixels):
    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, w, h)

    roi = depth_map[y1:y2, x1:x2]
    valid = roi[np.isfinite(roi)]
    valid = valid[(valid > min_valid) & (valid < max_valid)]

    if valid.size < min_pixels:
        return None

    return float(np.median(valid))


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


def midas_to_meters(depth_norm, near_m, far_m):
    return near_m + (1.0 - depth_norm) * (far_m - near_m)


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
        description="Compare subtask2 hyperparameters with timing and distance stats."
    )
    parser.add_argument("--left-dir", default="images/left", help="Left image folder")
    parser.add_argument("--output-dir", default="outputs", help="Output folder")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--yolo-model", default="yolo26m.pt", help="YOLO model path")
    parser.add_argument("--config", default="", help="JSON file of hyperparam configs")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    left_dir = base_dir / args.left_dir
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not left_dir.exists():
        raise FileNotFoundError("Left image folder not found.")

    image_paths = list_images(left_dir)
    if not image_paths:
        raise FileNotFoundError("No images found in left folder.")

    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        device = torch.device("cpu")

    midas, transform, midas_load_s = load_midas(device)

    yolo_load_start = time.perf_counter()
    yolo = YOLO(args.yolo_model)
    if hasattr(yolo, "to"):
        yolo.to(device)
    sync_if_cuda(device)
    yolo_load_s = time.perf_counter() - yolo_load_start

    configs = load_configs(args.config)
    rows = []

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"Skipping {image_path.name}: could not read image.")
            continue

        sync_if_cuda(device)
        t0 = time.perf_counter()
        midas_depth = midas_depth_map_from_bgr(image_bgr, midas, transform, device)
        sync_if_cuda(device)
        t1 = time.perf_counter()
        time_midas_s = t1 - t0

        depth_norm = normalize_map(midas_depth)

        for cfg in configs:
            sync_if_cuda(device)
            t2 = time.perf_counter()
            results = yolo(image_bgr, conf=cfg["score_thresh"], verbose=False)[0]
            sync_if_cuda(device)
            t3 = time.perf_counter()
            time_yolo_s = t3 - t2

            t4 = time.perf_counter()
            depth_m = midas_to_meters(depth_norm, cfg["near_m"], cfg["far_m"])

            boxes = results.boxes
            distances = []
            num_detections = 0

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                num_detections = xyxy.shape[0]
                for box in xyxy:
                    distance = estimate_distance_from_depth(
                        depth_m,
                        box,
                        cfg["min_valid"],
                        cfg["max_valid"],
                        cfg["min_pixels"],
                    )
                    if distance is not None:
                        distances.append(distance)

            time_post_s = time.perf_counter() - t4

            if distances:
                dist_array = np.array(distances, dtype=np.float32)
                mean_distance = float(np.mean(dist_array))
                median_distance = float(np.median(dist_array))
                std_distance = float(np.std(dist_array))
                min_distance = float(np.min(dist_array))
                max_distance = float(np.max(dist_array))
            else:
                mean_distance = float("nan")
                median_distance = float("nan")
                std_distance = float("nan")
                min_distance = float("nan")
                max_distance = float("nan")

            num_valid = len(distances)
            frac_valid = num_valid / num_detections if num_detections else 0.0

            rows.append(
                {
                    "image": image_path.name,
                    "config": cfg["name"],
                    "score_thresh": cfg["score_thresh"],
                    "near_m": cfg["near_m"],
                    "far_m": cfg["far_m"],
                    "min_valid": cfg["min_valid"],
                    "max_valid": cfg["max_valid"],
                    "min_pixels": cfg["min_pixels"],
                    "num_detections": num_detections,
                    "num_valid": num_valid,
                    "frac_valid": frac_valid,
                    "mean_distance_m": mean_distance,
                    "median_distance_m": median_distance,
                    "std_distance_m": std_distance,
                    "min_distance_m": min_distance,
                    "max_distance_m": max_distance,
                    "time_midas_s": time_midas_s,
                    "time_yolo_s": time_yolo_s,
                    "time_post_s": time_post_s,
                    "time_total_s": time_midas_s + time_yolo_s + time_post_s,
                    "depth_cached": 1,
                }
            )

    csv_path = output_dir / "subtask2_hyperparam_metrics.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "device": str(device),
        "images": len(set(r["image"] for r in rows)),
        "midas_model_load_s": midas_load_s,
        "yolo_model_load_s": yolo_load_s,
        "depth_cached": True,
        "configs": {},
        "csv": str(csv_path),
    }

    by_config = {}
    for row in rows:
        by_config.setdefault(row["config"], []).append(row)

    for cfg_name, items in by_config.items():
        summary["configs"][cfg_name] = {
            "images": len(items),
            "avg_detections": mean([r["num_detections"] for r in items]),
            "avg_valid": mean([r["num_valid"] for r in items]),
            "avg_frac_valid": mean([r["frac_valid"] for r in items]),
            "avg_mean_distance_m": mean([r["mean_distance_m"] for r in items]),
            "avg_time_total_s": mean([r["time_total_s"] for r in items]),
            "avg_time_yolo_s": mean([r["time_yolo_s"] for r in items]),
            "avg_time_midas_s": mean([r["time_midas_s"] for r in items]),
            "avg_time_post_s": mean([r["time_post_s"] for r in items]),
        }

    summary_rows = []
    for cfg_name, items in by_config.items():
        if not items:
            continue
        first = items[0]
        summary_rows.append(
            {
                "config": cfg_name,
                "score_thresh": first["score_thresh"],
                "near_m": first["near_m"],
                "far_m": first["far_m"],
                "min_valid": first["min_valid"],
                "max_valid": first["max_valid"],
                "min_pixels": first["min_pixels"],
                "images": len(items),
                "avg_detections": mean([r["num_detections"] for r in items]),
                "avg_valid": mean([r["num_valid"] for r in items]),
                "avg_frac_valid": mean([r["frac_valid"] for r in items]),
                "avg_mean_distance_m": mean([r["mean_distance_m"] for r in items]),
                "avg_median_distance_m": mean([r["median_distance_m"] for r in items]),
                "avg_time_total_s": mean([r["time_total_s"] for r in items]),
                "avg_time_yolo_s": mean([r["time_yolo_s"] for r in items]),
                "avg_time_midas_s": mean([r["time_midas_s"] for r in items]),
                "avg_time_post_s": mean([r["time_post_s"] for r in items]),
            }
        )

    summary_csv_path = output_dir / "subtask2_hyperparam_summary.csv"
    summary_fields = list(summary_rows[0].keys()) if summary_rows else []
    with open(summary_csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary["csv_summary"] = str(summary_csv_path)

    summary_path = output_dir / "subtask2_hyperparam_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved metrics to {csv_path}")
    print(f"Saved summary table to {summary_csv_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
