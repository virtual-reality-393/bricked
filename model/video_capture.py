import cv2
import numpy as np
from mss import mss
import PIL.Image as pil
from PIL import Image
from torchvision import datasets, transforms
import torch
import matplotlib as mpl
from brick import *
import pygetwindow as gw
import matplotlib.pyplot as plt

class_names = ["brick"]
def normalize(data):
    return (data - data.min()) / (data.max() - data.min())

detection = BrickDetector()
# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [
        base_colors[color_index][i]
        + increments[color_index][i] * (cls_num // len(base_colors)) % 256
        for i in range(3)
    ]
    return tuple(color)

cv2.namedWindow("frame")
cv2.moveWindow("frame",0,0)
with mss() as sct:
    while True:

        
        # monitor = sct.monitors[2]
        window = gw.getWindowsWithTitle("Casting")[0]
        # monitor = window.left+25, window.top+100, window.left + window.width-25, window.top + window.height-75
        monitor = window.left + window.width //2 -  window.height//2, window.top, window.left +window.width//2 + window.height//2, window.top + window.height

        
        
        sct_img = sct.grab(monitor)

        frame = np.array(sct_img)
        frame = cv2.resize(frame,(1024,1024))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        bboxes,bboxes2 = detection.detect(frame,conf=0.4)

        for box in bboxes:
            # check if confidence is greater than 40 percent
            if box.conf[0] > 0.4:
                # get coordinates
                [x,y,w,h] = box.xywh[0]

                x1 = x - w/2
                x2 = x + w/2
                y1 = y - h/2
                y2 = y + h/2
                # convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # get the class
                cls = int(box.cls[0])

                # get the class name
                class_name = class_names[cls]

                # get the respective colour
                colour = (0,255,0)

                # draw the rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                # put the class name and confidence on the image
                cv2.putText(frame, f'{class_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

        for box in bboxes2:
            # check if confidence is greater than 40 percent
            if box.conf[0] > 0.4:
                # get coordinates
                [x,y,w,h] = box.xywh[0]

                x1 = x - w/2
                x2 = x + w/2
                y1 = y - h/2
                y2 = y + h/2
                # convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # get the class
                cls = int(box.cls[0])

                # get the class name
                class_name = class_names[cls]

                # get the respective colour
                colour = (0,0,255)

                # draw the rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                # put the class name and confidence on the image
                cv2.putText(frame, f'{class_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

        # show the image
        cv2.imshow("frame", frame)

        # break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break
