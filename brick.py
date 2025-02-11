import cv2
import numpy as np
from ultralytics import YOLO
import ultralytics.engine.results
import random

yolo_model = YOLO("yolo11s.pt", verbose=False)

import matplotlib.pyplot as plt



def random_color():
    return np.random.randint(0, 255, (3), dtype=np.uint8).tolist()


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)

    # Add any transformation here
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def detect(image: np.ndarray, conf: float = 0.4) -> ultralytics.engine.results.Results:
    results = yolo_model(image)[0]
    return results


def draw_box(image, xywh):
    x,y,w,h = xywh

    x1 = x
    x2 = x+w
    y1 = y
    y2 = y+h

    cv2.rectangle(
        image, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=random_color(),thickness=10
    )
