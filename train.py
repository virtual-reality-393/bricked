from ultralytics import YOLO

model = YOLO("yolo11s.pt")  # Load a pretrained model
results = model.train(data="data.yaml", epochs=1000, imgsz=640,workers = 0)