import pygetwindow as gw
from PIL import Image
from io import BytesIO
import glob
from pathlib import Path
import os
import cv2

from brick import *


frame_num = 0
imgs = sorted(glob.glob("color_processed_data/*.jpg"))
labels = sorted(glob.glob("color_processed_data/*.txt"))


img_labels = list(zip(imgs,labels))
img_labels = sorted(img_labels,key = lambda e: os.path.getsize(e[1]))

def draw_box_color(image, xywh,color):
    x,y,w,h = xywh

    x1 = x
    x2 = x+w
    y1 = y
    y2 = y+h

    cv2.rectangle(
        image, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=color,thickness=2
    )


for img_path,label_path in img_labels:
    image = cv2.imread(img_path)
    labels = read_file(label_path) 
    
    for line in labels:
        label, x, y, w, h = line.split(" ")

        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)

        x1 = x-w/2
        y1 = y-h/2
        
        draw_box_color(image,(x1*image.shape[1],y1*image.shape[0],w*image.shape[1],h*image.shape[0]),(0,255,0))

    image = cv2.resize(image,(640,640))
    print(img_path)
    while True:
        cv2.imshow("img",image)

        key = cv2.waitKey()
        if key == ord("q"):
            exit()
        elif key == ord("z"):
            os.remove(img_path)
            os.remove(label_path)
        else:
            break








