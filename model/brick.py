import cv2
import numpy as np
from ultralytics import YOLO
import ultralytics.engine.results
import random
import matplotlib.pyplot as plt

yolo_model = None




def random_color():
    return np.random.randint(0, 255, (3), dtype=np.uint8).tolist()


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)

    # Add any transformation here
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def detect(image: np.ndarray, conf: float = 0.3, is_video = False):
    global yolo_model
    if yolo_model == None:
        yolo_model = YOLO(r"C:\Users\VirtualReality\Desktop\bricked\runs\detect\train15\weights\best.pt", verbose=True)
    results = yolo_model.track(image,stream=is_video,persist=is_video)
    bboxes = []
    try:
        for result in results:
            for box in result.boxes:
                if box.conf > conf:
                    bboxes.append(box)
        
        return bboxes
    except:
        return bboxes
    
def annotate_image(image,bboxes,class_names = ["brick"], colors = [(1,0,0),(0,1,0),(0,0,1)]):
    for box in bboxes:
        # check if confidence is greater than 40 percent
        if box.conf[0] > 0.4:
            # get coordinates
            [x,y,w,h] = box.xywh[0]

            x1 = x
            x2 = x + w
            y1 = y
            y2 = y + h
            # convert to int
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # get the class
            cls = int(box.cls[0])

            # get the class name
            class_name = class_names[cls]

            # get the respective color
            color = colors[cls]

            # draw the rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # put the class name and confidence on the image
            cv2.putText(image, f'{class_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

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