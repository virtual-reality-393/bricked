import cv2
import numpy as np
import cv2
from mss import mss
import cv2
import PIL.Image as pil
from PIL import Image
from torchvision import datasets, transforms
import torch
import matplotlib as mpl


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


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


with mss() as sct:
    while True:
        monitor = sct.monitors[2]
        sct_img = sct.grab(monitor)

        frame = np.array(sct_img)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        # show the image
        cv2.imshow("frame", frame)

        # break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break
