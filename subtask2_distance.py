from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import midas_utils


LEFT_DIR = "images/left"
OUTPUT_DIR = "outputs"
DEVICE = "cuda"
YOLO_MODEL = "yolo26m.pt"
SCORE_THRESH = 0.7
NEAR_M = 1.0
FAR_M = 50.0
MIN_VALID = 0.1
MAX_VALID = 200.0
MIN_PIXELS = 25


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


def midas_to_meters(midas_depth):
    depth_norm = normalize_map(midas_depth)
    return NEAR_M + (1.0 - depth_norm) * (FAR_M - NEAR_M)


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


def estimate_distance_from_depth(depth_map, box):
    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, w, h)

    roi = depth_map[y1:y2, x1:x2]
    valid = roi[np.isfinite(roi)]
    valid = valid[(valid > MIN_VALID) & (valid < MAX_VALID)]

    if valid.size < MIN_PIXELS:
        return None

    return float(np.median(valid))


def color_for_label(label_id):
    palette = [
        (255, 99, 71),
        (30, 144, 255),
        (50, 205, 50),
        (255, 215, 0),
        (138, 43, 226),
        (255, 140, 0),
        (0, 206, 209),
    ]
    return palette[int(label_id) % len(palette)]


def annotate_image(image_bgr, detections):
    annotated = image_bgr.copy()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        distance = det["distance"]
        color = det["color"]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        if distance is None:
            text = f"{label}: n/a"
        else:
            text = f"{label}: {distance:.1f} m"

        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            annotated,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w + 4, y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            text,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated


def main():
    base_dir = Path(__file__).resolve().parent
    left_dir = base_dir / LEFT_DIR
    output_dir = base_dir / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not left_dir.exists():
        raise FileNotFoundError("Left image folder not found.")

    image_paths = list_images(left_dir)
    if not image_paths:
        raise FileNotFoundError("No images found in left folder.")

    device = torch.device(DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        device = torch.device("cpu")

    yolo = YOLO(YOLO_MODEL)
    if hasattr(yolo, "to"):
        yolo.to(device)

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"Skipping {image_path.name}: could not read image.")
            continue

        midas_depth = midas_utils.midas_depth_map(str(image_path), device)
        depth_map = midas_to_meters(midas_depth)

        results = yolo(image_bgr, conf=SCORE_THRESH, verbose=False)[0]
        boxes = results.boxes
        names = results.names if hasattr(results, "names") else getattr(yolo, "names", {})

        detections = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            for box, cls_id in zip(xyxy, cls_ids):
                if isinstance(names, dict):
                    label = names.get(cls_id, str(cls_id))
                else:
                    label = str(names[cls_id])
                distance = estimate_distance_from_depth(depth_map, box)
                color = color_for_label(cls_id)
                x1, y1, x2, y2 = clamp_box(box, image_bgr.shape[1], image_bgr.shape[0])
                detections.append(
                    {
                        "box": (x1, y1, x2, y2),
                        "label": label,
                        "distance": distance,
                        "color": color,
                    }
                )

        annotated = annotate_image(image_bgr, detections)
        output_path = output_dir / f"{image_path.stem}_yolo_midas.png"
        cv2.imwrite(str(output_path), annotated)
        print(f"Saved {output_path.name} ({len(detections)} detections)")


if __name__ == "__main__":
    main()
