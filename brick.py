import cv2
import numpy as np
from ultralytics import YOLO
import ultralytics.engine.results
import random
import matplotlib.pyplot as plt

yolo_model = YOLO(r"C:\Users\VirtualReality\Desktop\bricked\runs\detect\train5\weights\best.pt", verbose=False)




def random_color():
    return np.random.randint(0, 255, (3), dtype=np.uint8).tolist()


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)

    # Add any transformation here
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def detect(image: np.ndarray, conf: float = 0.1):
    results = yolo_model.track(image,stream=True,persist=True)
    result_data = []
    for result in results:
        bboxes = []
        for box in result.boxes:
            if box.conf > conf:
                bboxes.append(box)
    
    return bboxes


def draw_box(image, xywh):
    x,y,w,h = xywh

    x1 = x
    x2 = x+w
    y1 = y
    y2 = y+h

    cv2.rectangle(
        image, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=random_color(),thickness=5
    )

def write_file(fn,content):
    with open(fn,"w") as text_file:
        text_file.write(content)

def read_file(fn):
    contents = ""
    with open(fn,"r") as text_file:
        contents = text_file.readlines()
    return contents