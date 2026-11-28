import io
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from ultralytics import YOLO

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8n.pt")
IMAGE_SIZE = 640                 # Higher than the original 416
TILE_SIZE = 640                  # Pixels per crop
TILE_OVERLAP = 0.20              # 20% overlap prevents objects being cut at edges
CONFIDENCE = 0.15                # Lower threshold helps retain small/faint objects
NMS_IOU = 0.50
MAX_DETECTIONS = 500
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_TILES = 36                   # Safety limit for very large uploads
DEVICE = 0 if torch.cuda.is_available() else "cpu"
DEVICE_NAME = "CUDA GPU" if torch.cuda.is_available() else "CPU"

app = FastAPI(title="VisionPort - Indian Traffic Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "null",
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"Loading {MODEL_PATH} on {DEVICE_NAME}...")
model = YOLO(MODEL_PATH)
print("YOLO model loaded successfully!")


def tile_positions(length: int, tile_size: int, overlap: float) -> List[int]:
    """Return crop start positions, including a final crop aligned to the edge."""
    if length <= tile_size:
        return [0]
    stride = max(1, int(tile_size * (1 - overlap)))
    positions = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def detect_on_image(image: np.ndarray) -> np.ndarray:
    """Run overlapping-tile detection, merge duplicate boxes, and draw results."""
    height, width = image.shape[:2]
    xs = tile_positions(width, TILE_SIZE, TILE_OVERLAP)
    ys = tile_positions(height, TILE_SIZE, TILE_OVERLAP)

    # Avoid unexpectedly long processing on enormous images.
    if len(xs) * len(ys) > MAX_TILES:
        scale = (MAX_TILES / (len(xs) * len(ys))) ** 0.5
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        height, width = image.shape[:2]
        xs = tile_positions(width, TILE_SIZE, TILE_OVERLAP)
        ys = tile_positions(height, TILE_SIZE, TILE_OVERLAP)

    all_boxes: List[List[int]] = []
    all_scores: List[float] = []
    all_classes: List[int] = []

    for y in ys:
        for x in xs:
            crop = image[y:min(y + TILE_SIZE, height), x:min(x + TILE_SIZE, width)]
            if crop.size == 0:
                continue

            results = model.predict(
                source=crop,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                iou=NMS_IOU,
                device=DEVICE,
                max_det=MAX_DETECTIONS,
                verbose=False,
            )

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)

            for box, score, class_id in zip(xyxy, scores, classes):
                x1, y1, x2, y2 = box
                # Convert crop-relative coordinates to original-image coordinates.
                all_boxes.append([
                    int(x1 + x), int(y1 + y),
                    int(x2 + x), int(y2 + y)
                ])
                all_scores.append(float(score))
                all_classes.append(int(class_id))

    # Class-wise NMS avoids suppressing different object categories.
    kept_indices = []
    if all_boxes:
        boxes_xywh = [
            [b[0], b[1], max(1, b[2] - b[0]), max(1, b[3] - b[1])]
            for b in all_boxes
        ]
        for class_id in set(all_classes):
            class_indices = [i for i, c in enumerate(all_classes) if c == class_id]
            local = cv2.dnn.NMSBoxes(
                [boxes_xywh[i] for i in class_indices],
                [all_scores[i] for i in class_indices],
                CONFIDENCE,
                NMS_IOU,
            )
            if len(local) > 0:
                kept_indices.extend(class_indices[int(i)] for i in np.array(local).flatten())

    annotated = image.copy()
    for index in kept_indices:
        x1, y1, x2, y2 = all_boxes[index]
        class_id = all_classes[index]
        score = all_scores[index]
        label = model.names.get(class_id, str(class_id))
        color = (40, 180, 80)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_y = max(y1, text_h + 8)
        cv2.rectangle(
            annotated,
            (x1, label_y - text_h - 8),
            (x1 + text_w + 6, label_y + baseline - 4),
            color,
            -1,
        )
        cv2.putText(
            annotated, text, (x1 + 3, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            cv2.LINE_AA,
        )

    print(f"Detection complete: {len(kept_indices)} objects, {len(xs) * len(ys)} tiles")
    return annotated


@app.get("/")
async def home():
    return {
        "status": "Active",
        "message": "VisionPort Indian Traffic Detection API is running!",
        "model": MODEL_PATH,
        "device": DEVICE_NAME,
    }


@app.get("/health")
async def health():
    return {
        "status": "Active",
        "model": MODEL_PATH,
        "device": DEVICE_NAME,
        "image_size": IMAGE_SIZE,
        "tile_size": TILE_SIZE,
        "tile_overlap": TILE_OVERLAP,
        "confidence_threshold": CONFIDENCE,
    }


@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large. Maximum size is 20 MB.")

    image_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        annotated = await run_in_threadpool(detect_on_image, image)
        success, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise HTTPException(status_code=500, detail="Image encoding failed.")

        return StreamingResponse(
            io.BytesIO(encoded.tobytes()),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        print("Detection Error:", str(exc))
        raise HTTPException(status_code=500, detail="Detection failed. Check the backend terminal.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
