import sys
import os

# Add the directory containing brick.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))


import cv2
import os
import glob
from pathlib import Path
import numpy as np
from model.brick import *

def get_color_name(h,s,v):
    h = int(h)
    h = h*2

    if v < 50:
        return "black"
    elif v > 200 and s < 30:
        return "white"
    elif h < 30 or h > 330:
        return "red"
    elif 30 <= h < 90:
        return "yellow"
    elif 90 <= h < 150:
        return "green"
    elif 150 <= h < 210:
        return "cyan"
    elif 210 <= h < 270:
        return "blue"
    elif 270 <= h < 330:
        return "magenta"
    else:
        return "unknown"


def find_brick_by_color(image, color_name, is_video=False):
    bboxes = detect(image, conf=0.4,is_video=is_video)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for box in bboxes:
        [x, y, w, h] = box.xywh[0]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        
        w10 = int(w * 0.3)
        h10 = int(h * 0.3)
        x1_new = x1 + int(w * 0.35)
        y1_new = y1 + int(h * 0.35)
        x2_new = x1_new + w10
        y2_new = y1_new + h10

        # get average hsv value of the bounding box
        hsv = cv2.cvtColor(image[y1_new:y2_new, x1_new:x2_new], cv2.COLOR_RGB2HSV)
        hsv = np.median(hsv,axis=(0,1))
        h,s,v = hsv
        detected_color_name = get_color_name(h,s,v)
        if detected_color_name.lower() == color_name.lower():
            return (x1, y1, x2, y2)
            
    return None

def find_brick_and_color(image, is_video=False):
    bboxes = detect(image, conf=0.4,is_video=is_video)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = []
    for box in bboxes:
        [x, y, w, h] = box.xywh[0]
        x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x + w/2), int(y + h/2)
        
        w10 = int(w * 0.3)
        h10 = int(h * 0.3)
        x1_new = x1 + int(w * 0.35)
        y1_new = y1 + int(h * 0.35)
        x2_new = x1_new + w10
        y2_new = y1_new + h10

        # get average hsv value of the bounding box
        hsv = cv2.cvtColor(image[y1_new:y2_new, x1_new:x2_new], cv2.COLOR_RGB2HSV)
        hsv = np.median(hsv,axis=(0,1))
        h,s,v = hsv
        detected_color_name = get_color_name(h,s,v)
        res.append((detected_color_name,(x1, y1, x2, y2)))
            
    return res

def color_of_pixel(image, x, y):

    hsv = hsv[0][0]
    h,s,v = hsv
    return hsv,get_color_name(h,s,v)

def hsv_to_rgb(h, s, v):
    hsv = np.uint8([[[h, s, v]]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
    return tuple(int(c) for c in rgb)

def process_video():

    vid_path = "bricked/exercise/video/test_video.mp4"
    self = "bricked/exercise/TestImages/"

    img_idx = 0

    frames_this_vid = 0
    print(f"Processing {vid_path}")
    video_capture = cv2.VideoCapture(vid_path)
    saved_frame_name = 0
    while video_capture.isOpened():
        frame_is_read, frame = video_capture.read()

        if frame_is_read:
            if saved_frame_name % 60 == 0:
                cv2.imwrite(f"{self}{str(img_idx)}.jpg", frame)
                img_idx+=1
                frames_this_vid+=1
            saved_frame_name += 1
        else:
            break
    print(f"Saved {frames_this_vid} images from {vid_path}")
    video_capture.release()


    print(f"Saved {img_idx} images in total")


if __name__ == "__main__":
    process_video()