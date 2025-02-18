import sys
import os

# Add the directory containing brick.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))



import cv2
import numpy as np
from mss import mss
import PIL.Image as pil
from PIL import Image
from torchvision import datasets, transforms
import torch
import matplotlib as mpl
from brick import *
from util import *
import pygetwindow as gw


class_names = ["brick"]
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


# image = Image.open("bricked/exercise/TestImages/6.jpg")
# image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
# image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

# colors = ["red","green","blue","yellow"]
# for color in colors:
#     color_name_to_find = color
#     brick_coordinates = find_brick_by_color(image, color_name_to_find, is_video=True)

#     if brick_coordinates:
#         x1, y1, x2, y2 = brick_coordinates
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(image, f'{color_name_to_find} brick', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#     else:
#         print(f"No {color_name_to_find} brick found in the image.")


# # show the image
# cv2.imshow("frame", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

name_to_color = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0), "yellow": (0, 255, 255)}
video = cv2.VideoCapture("bricked/exercise/Video/20250218_091630_a54225f8.mp4")
while video.isOpened():
        frame_is_read, frame = video.read()

        frame = cv2.resize(frame,(frame.shape[1]//2,frame.shape[0]//2))

        colors = ["red","green","blue","yellow"]
        for color in colors:
            color_name_to_find = color
            brick_coordinates = find_brick_by_color(frame, color_name_to_find, is_video=True)

            if brick_coordinates:
                x1, y1, x2, y2 = brick_coordinates
                cv2.rectangle(frame, (x1, y1), (x2, y2), name_to_color[color], 2)
                cv2.putText(frame, f'{color_name_to_find} brick', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, name_to_color[color], 2)
            else:
                print(f"No {color_name_to_find} brick found in the image.")

        if frame_is_read:
            cv2.imshow("frame", frame)


            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                break
        else:
            break


# bboxes = detect(image,is_video=False)

# num_briks = 0
# num_tagets = 0

# for box in bboxes:
#     # check if confidence is greater than 40 percent
#     if box.conf[0] > 0.4:
#         # get coordinates
#         [x,y,w,h] = box.xywh[0]

#         x1 = x
#         x2 = x + w
#         y1 = y
#         y2 = y + h
#         # convert to int
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#         #get pixel color in the center of the bounding box
#         temnp = image[y1 + int(h)//2, x1 + int(w)//2]
#         colour = (int(temnp[0]), int(temnp[1]), int(temnp[2]))
#         print(colour)

#         # get the class
#         cls = int(box.cls[0])


#         # draw the rectangle
#         cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)

#         colour_name = get_color_name(colour)

#         # draw the center of the bounding box
#         #cv2.circle(image, (x1 + int(w)//2, y1 + int(h)//2), 5, (255,255,255), -1)

#         # put the class name and confidence on the image
#         cv2.putText(image, f'{colour_name + class_names[int(box.cls[0])]}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
#         #cv2.putText(image, f'{class_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

# # show the image
# cv2.imshow("frame", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# with mss() as sct:
#     while True:
#         # monitor = sct.monitors[2]
#         window = gw.getWindowsWithTitle("Pixel 6 Pro")[0]
#         # monitor = window.left+25, window.top+100, window.left + window.width-25, window.top + window.height-75
#         monitor = window.left, window.top, window.left + window.width, window.top + window.height
        
#         sct_img = sct.grab(monitor)

#         frame = np.array(sct_img)

#         frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

#         bboxes = detect(frame,is_video=True)

#         for box in bboxes:
#             # check if confidence is greater than 40 percent
#             if box.conf[0] > 0.4:
#                 # get coordinates
#                 [x,y,w,h] = box.xywh[0]

#                 x1 = x
#                 x2 = x + w
#                 y1 = y
#                 y2 = y + h
#                 # convert to int
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#                 # get the class
#                 cls = int(box.cls[0])

#                 # get the class name
#                 class_name = class_names[cls]

#                 # get the respective colour
#                 colour = getColours(cls)

#                 # draw the rectangle
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

#                 # put the class name and confidence on the image
#                 cv2.putText(frame, f'{class_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

#         # show the image
#         cv2.imshow("frame", frame)

#         # break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#         if cv2.waitKey(1) == ord("q"):
#             cv2.destroyAllWindows()
#             break