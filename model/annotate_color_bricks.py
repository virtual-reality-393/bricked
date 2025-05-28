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

IN_FOLDER = "C:\\Users\\VirtualReality\\Desktop\\bricked\\model\\processed_data\\"
OUT_FOLDER = "C:\\Users\\VirtualReality\\Desktop\\bricked\\model\\color_processed_data\\"
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
    for (img,label) in tqdm(zipList):
        yield (cv2.imread(img),read_file(label),img,label)


train_img_paths = glob.glob(IN_FOLDER + "*.jpg",recursive=True)
train_label_paths = glob.glob(IN_FOLDER + "*.txt",recursive=True)


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
        bricks_box.append([label[0],x,y,w,h])


    new_labels = []

    for box in bricks_box:
        [class_label,x, y, w, h] = box

        new_labels.append(f"{class_label.split('.0')[0]} {x/img.shape[1]} {y/img.shape[0]} {w/img.shape[1]} {h/img.shape[0]}")
        


    # print(label_path.split("\\"))

    backslash = "\\"

    cv2.imwrite(f"{OUT_FOLDER}{img_path.split(backslash)[-1]}",img)
    write_file(f"{OUT_FOLDER}{label_path.split(backslash)[-1]}","\n".join(new_labels))

