from ultralytics import YOLO
import os
MODEL = "models/run54_separate_nano.pt"
model = YOLO(MODEL,verbose=True)  # Load a pretrained model
model.export(format="mnn",int8=True)