import cv2
import numpy as np
from mss import mss
import PIL.Image as pil
from PIL import Image
from torchvision import datasets, transforms
import torch
import matplotlib as mpl
#from brick import *
from util import *
import pygetwindow as gw


brick_centers = {}
target = None
brick_to_move = None
task_completed = False
dist_to_target = {}
bricks_in_frame = {"red": False, "green": False, "blue": False, "yellow": False}
name_to_color = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 100, 100), "yellow": (0, 255, 255), "black": (0, 0, 0), "white": (255, 255, 255), "cyan": (255, 255, 0), "magenta": (255, 0, 255), "unknown": (255, 255, 255)}

# test_images_dir = "exercise/TestImages"
# for filename in os.listdir(test_images_dir):
#     if filename.endswith(".jpg") or filename.endswith(".png"):
#         image_path = os.path.join(test_images_dir, filename)
#         print(filename)
#         frame = Image.open(image_path)
#         frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
#         frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

#         brick_coordinates = find_brick_and_color(frame, is_video=True)

#         new_frame = draw_stack_box(frame, brick_coordinates, name_to_color)

#         # Show the image
#         cv2.imshow("frame", new_frame)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

frame = Image.open("exercise/TestImages/24.jpg")
frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

brick_coordinates = find_brick_and_color(frame, is_video=True)

new_frame,_ = draw_stack_box(frame, brick_coordinates, name_to_color)

# new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2HSV)

# cv2.putText(frame, f'Target: {target}', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
# cv2.putText(frame, f'brick to move: {brick_to_move}', (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

# show the image
cv2.imshow("frame", new_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


# video_path = "exercise/video/test_video.mp4"
# video = cv2.VideoCapture(filename=video_path)

# while video.isOpened():
#     frame_is_read, frame = video.read()

 
#     frame = cv2.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)))

#     brick_coordinates = find_brick_and_color(frame, is_video=True)

#     new_frame,_ = draw_stack_box(frame, brick_coordinates, name_to_color)

#     if frame_is_read:
#         cv2.imshow("frame", new_frame)


#         if cv2.waitKey(1) == ord("q"):
#             cv2.destroyAllWindows()
#             break
#     else:
#         break


