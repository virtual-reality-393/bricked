from ultralytics import YOLO

model = YOLO("yolo11s.pt")  # Load a pretrained model
results = model.train(data="data.yaml", epochs=100, imgsz=640,workers = 0)