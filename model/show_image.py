import pygetwindow as gw
from PIL import Image
from io import BytesIO
import glob
from pathlib import Path
import os
import cv2

from brick import *


frame_num = 0
imgs = (glob.glob("color_processed_data/*.jpg"))[-1:]
labels = (glob.glob("color_processed_data/*.txt"))[-1:]





img_labels = list(zip(imgs,labels))
img_labels = sorted(img_labels,key = lambda e: os.path.getsize(e[1]))
label_idx_to_name = {0:"red",1:"green",2:"blue",3:"yellow",4:"big penguin",5:"small penguin",6:"lion",7:"sheep",8:"pig",9:"human"}
def draw_box_color(image, xywh,color,cls,conf):
    x,y,w,h = xywh

    x1 = x
    x2 = x+w
    y1 = y
    y2 = y+h

    cv2.rectangle(
        image, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=color,thickness=2
    )

    cv2.putText(image, f"{label_idx_to_name[cls.item()]}: {conf.item():.3f}",(int(x1),int(y1)),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)

detector = BrickDetector(is_video=False)
label_idx_to_color = {0:(0,0,1,1),1:(0,1,0,1),2:(1,0,0,1),3:(0,1,1,1),4:(0.4,0.4,1,1),5:(0.4,0.4,0.4,1),6:(0,0.5,1,1),7:(1,1,1,1),8:(0.5,0.5,1,1),9:(1.0,0.2,0.8,1)}
for img_path,label_path in img_labels:
    image = cv2.imread("test.png")
    labels = read_file(label_path) 

    bboxes = detector.detect(image,model_to_use=1)

    
    for bbox in bboxes:
        x,y,w,h = bbox.xywh[0]

        

        x1 = x-w/2
        y1 = y-h/2
        x2 = x1+w
        y2 = y1+h

        color = tuple(int(a) for a in [num for num in np.array(label_idx_to_color[int(bbox.cls.item())])*255][:-1])

        draw_box_color(image,(x1,y1,w,h),color,bbox.cls,bbox.conf)
        
        

    image = cv2.resize(image,(640,640))

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








