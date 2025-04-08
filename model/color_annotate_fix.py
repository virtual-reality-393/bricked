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

index_to_name = {0:"red",1:"green",2:"blue",3:"yellow",4:"big penguin",5:"small penguin",6:"lion",7:"sheep",8:"pig",9:"human"}

index_to_color = {0:(1,0,0,1),1:(0,1,0,1),2:(0,0,1,1),3:(1,1,0,1),4:(0.4,0.4,1,1),5:(0.4,1,0.4,1),6:(1,1,0.4,1),7:(1,1,1,1),8:(1,0.5,0.5,1),9:(1.0,0.2,0.8,1)}


index_to_color = {k:(v[2],v[1],v[0],v[3]) for k,v in index_to_color.items()}


name_to_color = {index_to_name[k]:v for k, v in index_to_color.items()}
name_to_index = {v: k for k, v in index_to_name.items()}





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

def center_dist_calc(bbox,i,j):
    _,x1,y1,x2,y2 = bbox


    cx = x1+(x2-x1)/2
    cy = y1+(y2-y1)/2


    return np.linalg.norm(np.array([cx-i,cy-j]))
    
def load_entrypoint(zipList):

    for (img,label) in zipList:
        yield cv2.imread(img),read_file(label),img,label

def __on_click__(event,x,y,flags,param):
    global bboxes,scale_factor,annotation_class
    if event == cv2.EVENT_LBUTTONDOWN:
        i = int(x*scale_factor)
        j = int(y*scale_factor)
        

        sorted([bbox for bbox in bboxes if bbox[1] < i and bbox[2] < j and bbox[3] > i and bbox[4] > j],key = lambda bbox: center_dist_calc(bbox,i,j))[0][0] = index_to_name[annotation_class]

        

train_img_paths = glob.glob(OUT_FOLDER + "*.jpg",recursive=True)
train_label_paths = glob.glob(OUT_FOLDER + "*.txt",recursive=True)

train_paths = list(zip(train_img_paths,train_label_paths))[510:]

for img,label,imgpath,labelpath in load_entrypoint(train_paths):

    print(imgpath)
    bboxes = []
    org_img = img.copy()
    for oneLabel in label:
        labelParts = oneLabel.replace("\n","").split(" ")
        name = index_to_name[int(labelParts[0])]
        x = float(labelParts[1])*img.shape[1]
        y = float(labelParts[2])*img.shape[0]
        w = float(labelParts[3])*img.shape[1]
        h = float(labelParts[4])*img.shape[0]
        x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x + w/2), int(y + h/2)
        bboxes.append([name,x1,y1,x2,y2])
    cv2.namedWindow('image_display')

    cv2.setMouseCallback('image_display', __on_click__)
    while True:
        img = org_img.copy()
        

        for [name,x1,y1,x2,y2] in bboxes:
            test = tuple([int(col) for col in (np.array(name_to_color[name])[:3]*255).astype(np.uint8)]) # Ty cv2 i hate this
            cv2.rectangle(img,(x1,y1),(x2,y2),color=test,thickness=10)
        

        scale_factor = max((min(img.shape[:2]) / 1024.0),1)
        image_shape = (int(img.shape[1]/scale_factor), int(img.shape[0]/scale_factor))
        scaled_image = cv2.resize(img, image_shape)
        cv2.imshow("image_display", scaled_image)
        
        key_press = cv2.waitKey(1) & 0xff

        if key_press == ord("q"):
            exit()

        if key_press == ord("d"):
            break

        if key_press == ord("1"):
            annotation_class = 0
        if key_press == ord("2"):
            annotation_class = 1
        if key_press == ord("3"):
            annotation_class = 2
        if key_press == ord("4"):
            annotation_class = 3
    label_texts = []
    for [name,x1,y1,x2,y2] in bboxes:
        w = x2-x1
        h = y2-y1
        idx = name_to_index[name]
        label_texts.append(f"{idx} {(x1+w/2)/img.shape[1]} {(y1+h/2)/img.shape[0]} {(w)/img.shape[1]} {(h)/img.shape[0]}")


    write_file(labelpath,"\n".join(label_texts))

            
