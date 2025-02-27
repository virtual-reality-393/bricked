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

def find_order_of_stack(stack):
    bricks, _ = stack
    order = []
    res = []
    if len(bricks) == 1:
        return [bricks[0][0]]
    
    if len(bricks) == 2:
        return [bricks[0][0], bricks[1][0]]

    for i, brick in enumerate(bricks):
        color, center = brick
        dist1 = np.inf
        dist2 = np.inf
        id1 = 0
        id2 = 0
        vec1 = None
        vec2 = None
        for j,brick2 in enumerate(bricks):
            if brick != brick2:
                _, center2 = brick2
                dist = np.linalg.norm(np.array(center) - np.array(center2))
                if dist < dist1:
                    dist2 = dist1
                    dist1 = dist
                    vec2 = vec1
                    vec1 = np.array(center2) - np.array(center)
                    id2 = id1
                    id1 = j
                elif dist < dist2:
                    dist2 = dist
                    vec2 = np.array(center2) - np.array(center)
                    id2 = j
        # print(dist1, dist2)
        # print(vec1, vec2)
        if np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)) < 0:
            order.append((i,(id1,id2),False))
        else:
            order.append((i,(id1,None),True))
        
    start_id = next((brick for brick in order if brick[2]), None)
    if start_id == None:
        return []
    
    current_id = start_id[0]
    next_id = 0
    num_bricks_in_current_stack = 0

    id_order = []

    id_order.append(start_id[0])
    while num_bricks_in_current_stack < len(order):
        next_id1 = order[current_id][1][0]
        next_id2 = order[current_id][1][1]
        if next_id1 in id_order:
            next_id = next_id2
        elif next_id2 == None:
            next_id = next_id1
        else:
            next_id = next_id1
        if next_id == None:
            break
        id_order.append(next_id)
        current_id = next_id
        num_bricks_in_current_stack += 1
    
    for i in id_order:
        res.append(bricks[i][0])

    return res


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

# frame = Image.open("exercise/TestImages/24.jpg")
# frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
# frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

# brickDetector = BrickDetector()

# stacks, bricks, new_frame = find_stacks_and_bricks(frame, brickDetector,name_to_color)

# for stack in stacks:
#     order = find_order_of_stack(stack)
#     x1, y1, x2, y2 = stack[1]
#     for i, color in enumerate(order):
#         #Satck 
#         cv2.putText(new_frame, f'{color}', (x2+10, y2 - 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 12)
#         cv2.putText(new_frame, f'{color}', (x2+10, y2 - 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, name_to_color[color], 2)
#         #Revese Stack
#         cv2.putText(new_frame, f'{color}', (x1-70, y1 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 12)
#         cv2.putText(new_frame, f'{color}', (x1-70, y1 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, name_to_color[color], 2)

# # new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2HSV)

# # cv2.putText(frame, f'Target: {target}', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
# # cv2.putText(frame, f'brick to move: {brick_to_move}', (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

# # show the image
# cv2.imshow("frame", new_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

video_path = "exercise/video/test_video2.mp4"
video = cv2.VideoCapture(filename=video_path)
brickDetector = BrickDetector()
while video.isOpened():
    frame_is_read, frame = video.read()

    frame = cv2.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)))

    stacks, bricks, new_frame = find_stacks_and_bricks(frame, brickDetector)

    for j,stack in enumerate(stacks):
        order = find_order_of_stack(stack)
        x1, y1, x2, y2 = stack[1]
        if len(order) > 1:
            cv2.rectangle(new_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(new_frame, f'Stack {j}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            for i, color in enumerate(order):
                #Satck 
                cv2.putText(new_frame, f'{color}', (x2+10, y2 - 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 12)
                cv2.putText(new_frame, f'{color}', (x2+10, y2 - 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, name_to_color[color], 2)
                #Revese Stack
                cv2.putText(new_frame, f'{color}', (x1-70, y1 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 12)
                cv2.putText(new_frame, f'{color}', (x1-70, y1 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, name_to_color[color], 2)
        elif len(order) == 1:
            cv2.rectangle(new_frame, (x1, y1), (x2, y2), name_to_color[order[0]], 2)
            cv2.putText(new_frame, f'{order[0]} brick', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, name_to_color[order[0]], 2)

    if frame_is_read:
        cv2.imshow("frame", new_frame)


        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break
    else:
        break


