from ultralytics import YOLO
import os

if __name__ == "__main__":
    os.environ["YOLO_VERBOSE"] = "true"

    model = YOLO("yolo11n.pt",verbose=True)  # Load a pretrained model
    results = model.train(data="data_separate_color.yaml", epochs=1000, imgsz=640,workers = 8,patience = 50,deterministic = False)
