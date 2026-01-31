from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import uuid
import json
from datetime import datetime
from pathlib import Path

app = FastAPI(title="Cafe Table Usage Analysis")

# CORS — чтобы фронтенд мог обращаться к API из браузера
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # в продакшене укажите конкретный origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Paths ---
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
HISTORY_FILE = DATA_DIR / "history.json"

# --- Load model ---
model = YOLO("yolov8n.pt")  # pretrained COCO model

# --- Utils ---
def load_history():
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(record):
    history = load_history()
    history.append(record)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


# --- API ---
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(
            {"detail": "Не удалось прочитать изображение. Поддерживаются JPG, PNG и др."},
            status_code=400
        )

    try:
        results = model(image)[0]
    except Exception as e:
        return JSONResponse(
            {"detail": "Ошибка анализа: " + str(e)},
            status_code=500
        )

    tables = []
    people = []

    xyxy = results.boxes.xyxy if results.boxes is not None else []
    clss = results.boxes.cls if results.boxes is not None else []
    for box, cls in zip(xyxy, clss):
        cls = int(cls.item())
        x1, y1, x2, y2 = map(int, box.tolist())

        if model.names[cls] == "dining table":
            tables.append((x1, y1, x2, y2))
        elif model.names[cls] == "person":
            people.append((x1, y1, x2, y2))

    occupied = 0
    for table in tables:
        for person in people:
            if iou(table, person) > 0.1:
                occupied += 1
                break

    total_tables = len(tables)
    free_tables = total_tables - occupied
    occupancy_rate = round((occupied / total_tables) * 100, 2) if total_tables else 0

    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "total_tables": total_tables,
        "occupied_tables": occupied,
        "free_tables": free_tables,
        "occupancy_rate": occupancy_rate
    }

    save_history(record)

    return JSONResponse(record)


@app.get("/history")
def get_history():
    return load_history()


@app.get("/")
def root():
    """Отдаёт главную страницу фронтенда."""
    return FileResponse("frontend_index.html")


@app.get("/api/status")
def api_status():
    return {"status": "Backend is running"}
