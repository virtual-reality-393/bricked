from brick import *
import glob
import math
import matplotlib.pyplot as plt
import random
import os
from pathlib import Path
import glob
import math
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import shutil
from tqdm import tqdm

name_to_color = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0), "yellow": (0, 255, 255), "magenta": (255, 0, 255)}
name_to_index = {"red": 0, "green": 1, "blue": 2, "yellow": 3}
def __get_color_name__(h,s,v):
    h = int(h)
    h = h*2

    s = int(s) * 2
    v = int(v)*2


    # if v < 50:
    #     return "black"
    # elif v > 200 and s < 100:
    #     return "white"
    if h < 30 or h > 330:
        return "red"
    elif 30 <= h < 90:
        return "yellow"
    elif 90 <= h < 150:
        return "green"
    # elif 150 <= h < 210:
    #     return "cyan"
    elif 210 <= h < 270:
        return "blue"
    # elif 270 <= h < 330:
    #     return "magenta"
    else:
        return "magenta"


def load_entrypoint(zipList):

    for (img,label) in zipList:
        yield cv2.imread(img),read_file(label),img,label

train_img_paths = glob.glob("datasets/brick_separate/images/train/*.jpg",recursive=True)
val_img_paths = glob.glob("datasets/brick_separate/images/val/*.jpg",recursive=True)
train_label_paths = glob.glob("datasets/brick_separate/labels/train/*.txt",recursive=True)
val_label_paths = glob.glob("datasets/brick_separate/labels/val/*.txt",recursive=True)



train_paths = list(zip(train_img_paths,train_label_paths))

for (img,labels,img_path,label_path) in load_entrypoint(train_paths):

    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    bricks_box = []

    for label in labels:
        label = label.replace("\n","").split(" ")
        x = float(label[1])*img.shape[1]
        y = float(label[2])*img.shape[0]
        w = float(label[3])*img.shape[1]
        h = float(label[4])*img.shape[0]
        bricks_box.append([x,y,w,h])


    new_labels = []

    for box in bricks_box:
        [x, y, w, h] = box
        x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x + w/2), int(y + h/2)

        x1_new = max(0, int(x - w * 0.05))
        y1_new = max(0, int(y - h * 0.05))
        x2_new = min(img.shape[1], int(x + w * 0.05))
        y2_new = min(img.shape[0], int(y + h * 0.05))

        # Ensure the region is valid
        if x1_new >= x2_new or y1_new >= y2_new:
            continue

        # get average hsv value of the bounding box
        hsv = hsv_frame[y1_new:y2_new, x1_new:x2_new]
        hsv = np.median(hsv, axis=(0,1))
        hh, s, v = hsv
        detected_color_name = __get_color_name__(hh, s, v)

        if detected_color_name != "magenta":
            new_labels.append(f"{name_to_index[detected_color_name]} {x/img.shape[1]} {y/img.shape[0]} {w/img.shape[1]} {h/img.shape[0]}")


    # print(label_path.split("\\"))

    backslash = "\\"

    cv2.imwrite(f"datasets/brick_color_separate/images/train/{img_path.split(backslash)[1]}",img)
    write_file(f"datasets/brick_color_separate/labels/train/{label_path.split(backslash)[1]}","\n".join(new_labels))

val_paths = list(zip(val_img_paths,val_label_paths))

for (img,labels,img_path,label_path) in load_entrypoint(val_paths):

    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    bricks_box = []

    for label in labels:
        label = label.replace("\n","").split(" ")
        x = float(label[1])*img.shape[1]
        y = float(label[2])*img.shape[0]
        w = float(label[3])*img.shape[1]
        h = float(label[4])*img.shape[0]
        bricks_box.append([x,y,w,h])


    new_labels = []

    for box in bricks_box:
        [x, y, w, h] = box
        x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x + w/2), int(y + h/2)

        x1_new = max(0, int(x - w * 0.05))
        y1_new = max(0, int(y - h * 0.05))
        x2_new = min(img.shape[1], int(x + w * 0.05))
        y2_new = min(img.shape[0], int(y + h * 0.05))

        # Ensure the region is valid
        if x1_new >= x2_new or y1_new >= y2_new:
            continue

        # get average hsv value of the bounding box
        hsv = hsv_frame[y1_new:y2_new, x1_new:x2_new]
        hsv = np.median(hsv, axis=(0,1))
        hh, s, v = hsv
        detected_color_name = __get_color_name__(hh, s, v)

        if detected_color_name != "magenta":
            new_labels.append(f"{name_to_index[detected_color_name]} {x/img.shape[1]} {y/img.shape[0]} {w/img.shape[1]} {h/img.shape[0]}")


    # print(label_path.split("\\"))

    backslash = "\\"

    cv2.imwrite(f"datasets/brick_color_separate/images/val/{img_path.split(backslash)[1]}",img)
    write_file(f"datasets/brick_color_separate/labels/val/{label_path.split(backslash)[1]}","\n".join(new_labels))


