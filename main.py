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
from model.brick import *
from util import *
import pygetwindow as gw
from model.assistant import VoiceAssistant


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





# image = Image.open("bricked/exercise/TestImages/26.jpg")
# image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
# image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

# name_to_color = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0), "yellow": (0, 255, 255)}
# colors = ["red","green","blue","yellow"]
# for color in colors:
#     color_name_to_find = color
#     brick_coordinates = find_brick_by_color(image, color_name_to_find, is_video=True)

#     if brick_coordinates:
#         x1, y1, x2, y2 = brick_coordinates
#         cv2.rectangle(image, (x1, y1), (x2, y2), name_to_color[color], 2)
#         cv2.putText(image, f'{color_name_to_find} brick', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, name_to_color[color], 2)

#         center = (x1 + int((x2 - x1) / 2), y1 + int((y2 - y1) / 2))
#         brik_centers[color] = center
#         cv2.circle(image, (x1 + int((x2 - x1) / 2), y1 + int((y2 - y1) / 2)), 3, (255,255,255), -1)
#         cv2.putText(image, f'({x1 + int((x2 - x1) / 2)}, {y1 + int((y2 - y1) / 2)})', (x1 + int((x2 - x1) / 2), y1 + int((y2 - y1) / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

#         dist_to_target[color] = np.linalg.norm(np.array(brik_centers[target]) - np.array(brik_centers[color]))
#     else:
#         print(f"No {color_name_to_find} brick found in the image.")


# # show the image
# cv2.imshow("frame", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# target = "red"
# dist_to_target = {}
# name_to_color = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0), "yellow": (0, 255, 255)}
# video = cv2.VideoCapture("bricked/exercise/Video/20250218_091526_c486d487.mp4")
# while video.isOpened():
#     frame_is_read, frame = video.read()

#     frame = cv2.resize(frame,(frame.shape[1]//2,frame.shape[0]//2))

#     colors = ["red","green","blue","yellow"]
#     for color in colors:
#         color_name_to_find = color
#         brick_coordinates = find_brick_by_color(frame, color_name_to_find, is_video=True)

#         if brick_coordinates:
#             x1, y1, x2, y2 = brick_coordinates
#             cv2.rectangle(frame, (x1, y1), (x2, y2), name_to_color[color], 2)
#             cv2.putText(frame, f'{color_name_to_find} brick', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, name_to_color[color], 2)

#             center = (x1 + int((x2 - x1) / 2), y1 + int((y2 - y1) / 2))
#             brik_centers[color] = center
#             cv2.circle(frame, (x1 + int((x2 - x1) / 2), y1 + int((y2 - y1) / 2)), 3, (255,255,255), -1)
#             cv2.putText(frame, f'({x1 + int((x2 - x1) / 2)}, {y1 + int((y2 - y1) / 2)})', (x1 + int((x2 - x1) / 2), y1 + int((y2 - y1) / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

#             dist_to_target[color] = np.linalg.norm(np.array(brik_centers[target]) - np.array(brik_centers[color]))
#         else:
#             print(f"No {color_name_to_find} brick found in the image.")
        
#         if frame_is_read:
#             cv2.imshow("frame", frame)


#         if cv2.waitKey(1) == ord("q"):
#             cv2.destroyAllWindows()
#             break
#     else:
#         break



# brik_centers = {}
# brik_boxes = {}
# target = "red"
# dist_to_target = {}
# briks_in_frame = {"red": False, "green": False, "blue": False, "yellow": False}
# name_to_color = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0), "yellow": (0, 255, 255)}
# colors = ["red","green","blue","yellow"]

# with mss() as sct:
#     while True:
#         # monitor = sct.monitors[2]
#         window = gw.getWindowsWithTitle("Pixel 6 Pro")[0]
#         # monitor = window.left+25, window.top+100, window.left + window.width-25, window.top + window.height-75
#         monitor = window.left, window.top, window.left + window.width, window.top + window.height
        
#         sct_img = sct.grab(monitor)

#         frame = np.array(sct_img)

#         frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

#         for color in colors:
#             color_name_to_find = color
#             brick_coordinates = find_brick_by_color(frame, color_name_to_find, is_video=True)

#             if brick_coordinates:
#                 briks_in_frame[color] = True
#                 x1, y1, x2, y2 = brick_coordinates
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), name_to_color[color], 2)
#                 cv2.putText(frame, f'{color_name_to_find} brick', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, name_to_color[color], 2)

#                 center = (x1 + int((x2 - x1) / 2), y1 + int((y2 - y1) / 2))
#                 brik_centers[color] = center
#                 cv2.circle(frame, (x1 + int((x2 - x1) / 2), y1 + int((y2 - y1) / 2)), 3, (255,255,255), -1)
#                 cv2.putText(frame, f'({x1 + int((x2 - x1) / 2)}, {y1 + int((y2 - y1) / 2)})', (x1 + int((x2 - x1) / 2), y1 + int((y2 - y1) / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
               
#                 # if briks_in_frame[target]:
#                 #     dist_to_target[color] = np.linalg.norm(np.array(brik_centers[target]) - np.array(brik_centers[color]))
#             else:
#                 briks_in_frame[color] = False
#                 print(f"No {color_name_to_find} brick found in the image.")
#                 if color in brik_boxes:
#                     x1, y1, x2, y2 = brik_boxes[color]
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), name_to_color[color], 2)
#                     cv2.putText(frame, f'{color_name_to_find} brick (last known)', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, name_to_color[color], 2)
#                     center = brik_centers[color]
#                     cv2.circle(frame, center, 3, (255, 255, 255), -1)
#                     cv2.putText(frame, f'({center[0]}, {center[1]})', (center[0], center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


#         text = "Move blue to red"
#         lineColor = (255,255,255)
#         if briks_in_frame["blue"] and briks_in_frame["red"]:
#             dist_to_target['blue'] = np.linalg.norm(np.array(brik_centers[target]) - np.array(brik_centers['blue']))
#             if dist_to_target["blue"] < 100:
#                 text = "You so Good"
#                 lineColor = (0,255,0)
#         cv2.putText(frame, text, (160, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, lineColor, 2)

#         if briks_in_frame["blue"] and briks_in_frame["red"]:
#             cv2.line(frame, brik_centers["blue"], brik_centers["red"], lineColor, 2)

#         # show the image
#         cv2.imshow("frame", frame)

#         # break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#         if cv2.waitKey(1) == ord("q"):
#             cv2.destroyAllWindows()
#             break


if __name__ == "__main__":
    assistant = VoiceAssistant(voice = "model/models/joe")
    brik_centers = {}
    target = "red"
    brik_to_move = "blue"
    dist_to_target = {}
    briks_in_frame = {"red": False, "green": False, "blue": False, "yellow": False}
    name_to_color = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0), "yellow": (0, 255, 255)}
    #colors = ["red","green","blue","yellow"]
    
    text = f"Move {brik_to_move} to {target}"
    assistant.play_message(text)
    with mss() as sct:
        while True:
            # monitor = sct.monitors[2]
            window = gw.getWindowsWithTitle("Pixel 6 Pro")[0]
            # monitor = window.left+25, window.top+100, window.left + window.width-25, window.top + window.height-75
            monitor = window.left, window.top, window.left + window.width, window.top + window.height
            
            sct_img = sct.grab(monitor)

            frame = np.array(sct_img)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            brick_coordinates = find_brik_and_color(frame, is_video=True)

            briks_in_frame = {"red": False, "green": False, "blue": False, "yellow": False}
            for (color, box) in brick_coordinates:
                if color in briks_in_frame:
                    briks_in_frame[color] = True

                x1, y1, x2, y2 = box

                drawColor =(0,0,0)
                if color in name_to_color:
                    drawColor = name_to_color[color]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), drawColor, 2)
                cv2.putText(frame, f'{color} brick', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, drawColor, 2)

                center = (x1 + int((x2 - x1) / 2), y1 + int((y2 - y1) / 2))
                brik_centers[color] = center
                cv2.circle(frame, (x1 + int((x2 - x1) / 2), y1 + int((y2 - y1) / 2)), 3, (255,255,255), -1)
                cv2.putText(frame, f'({x1 + int((x2 - x1) / 2)}, {y1 + int((y2 - y1) / 2)})', (x1 + int((x2 - x1) / 2), y1 + int((y2 - y1) / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)


            
            lineColor = (255,255,255)
            if briks_in_frame[brik_to_move] and briks_in_frame[target]:
                dist_to_target[brik_to_move] = np.linalg.norm(np.array(brik_centers[target]) - np.array(brik_centers[brik_to_move]))
                if dist_to_target[brik_to_move] < 100:
                    text = "Good Job"
                    assistant.play_message(text)

                    lineColor = (0,255,0)
                    target = np.random.choice([color for color in briks_in_frame])
                    brik_to_move = np.random.choice([color for color in briks_in_frame if color != target])
                    text = f"Move {brik_to_move} to {target}"
                    assistant.play_message(text)

            cv2.putText(frame, text, (160, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, lineColor, 2)

            if briks_in_frame[brik_to_move] and briks_in_frame[target]:
                cv2.line(frame, brik_centers[brik_to_move], brik_centers[target], lineColor, 2)

            '''Displays hvilken klodser der er i frame'''
            # y_offset = 30
            # for color in briks_in_frame:
            #     if briks_in_frame[color]:
            #         cv2.putText(frame, f'{color} brick in frame', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, name_to_color[color], 2)
            #         y_offset += 20

            # show the image
            cv2.imshow("frame", frame)

            # break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                break