from ultralytics import YOLO

MODEL = "models/run64_stacks.pt"
model = YOLO(MODEL,verbose=True)  # Load a pretrained model
model.export(format="onnx")