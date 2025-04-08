from ultralytics import YOLO
import os
MODEL = "models/run59_figures.pt"
model = YOLO(MODEL,verbose=True)  # Load a pretrained model
model.export(format="onnx")