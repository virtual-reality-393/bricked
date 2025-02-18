from ultralytics import YOLO
import os

os.environ["YOLO_VERBOSE"] = "true"

model = YOLO("yolo11s.pt",verbose=True)  # Load a pretrained model
results = model.train(data="data.yaml", epochs=1000, imgsz=640,workers = 0)