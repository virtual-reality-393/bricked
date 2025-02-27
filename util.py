import cv2
import os
import glob
from pathlib import Path
import numpy as np
from model.brick import *
import math

def get_color_name(h,s,v):
    h = int(h)
    h = h*2

    s = int(s) * 2
    v = int(v)*2


    if v < 50:
        return "black"
    elif v > 200 and s < 100:
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

def stack_str_to_array(stack_str):
    if not stack_str:
        return []

    color_map = {
        'r': 'red',
        'y': 'yellow',
        'g': 'green',
        'b': 'blue'
    }

    stack_array = []
    current_color = stack_str[0]
    current_color_name = color_map[current_color]

    char_count = 1
    for char in stack_str[1:]:
        if char == current_color:
            char_count += 1
        else:
            if char_count > 1:
                stack_array.append((current_color_name, char_count))
            current_color = char
            current_color_name = color_map[current_color]
            char_count = 1

    # Append the last color if it appears more than 6 times
    if char_count > 1:
        stack_array.append((current_color_name, char_count))

    # Calculate the median char count
    counts = [count for _, count in stack_array]
    if counts:
        median_count = np.median(counts)
        stack_array = [(color, count) for color, count in stack_array if count > median_count * 0.7]

    # Remove consecutive duplicates
    filtered_stack_array = []
    prev_color = None
    for color, count in stack_array:
        if color != prev_color:
            filtered_stack_array.append((color, count))
        prev_color = color

    return filtered_stack_array

def centers_to_line(centers, box):
    # Calculate the parametric form of the line passing through the centers
    x1, y1, x2, y2 = box
    num_centers = len(centers)
    
    if num_centers < 2:
        return None

    # Initialize sums for averaging
    sum_x_start = 0
    sum_y_start = 0
    sum_x_end = 0
    sum_y_end = 0
    count = 0

    # Calculate lines between all pairs of centers
    for i in range(num_centers):
        for j in range(i + 1, num_centers):
            (x1_center, y1_center), (x2_center, y2_center) = centers[i], centers[j]
            dx = x2_center - x1_center
            dy = y2_center - y1_center

            # Calculate the intersection points with the box edges
            if dx != 0:
                t1 = (x1 - x1_center) / dx
                t2 = (x2 - x1_center) / dx
            else:
                t1 = float('-inf')
                t2 = float('inf')

            if dy != 0:
                t3 = (y1 - y1_center) / dy
                t4 = (y2 - y1_center) / dy
            else:
                t3 = float('-inf')
                t4 = float('inf')

            tmin = max(min(t1, t2), min(t3, t4))
            tmax = min(max(t1, t2), max(t3, t4))

            if tmax >= tmin and not (math.isnan(tmin) or math.isnan(tmax)):
                x_start = x1_center + tmin * dx
                y_start = y1_center + tmin * dy
                x_end = x1_center + tmax * dx
                y_end = y1_center + tmax * dy

                # Check for NaN values before proceeding
                if any(math.isnan(val) for val in [x_start, y_start, x_end, y_end]):
                    continue

                sum_x_start += int(x_start)
                sum_y_start += int(y_start)
                sum_x_end += int(x_end)
                sum_y_end += int(y_end)
                count += 1

    if count == 0:
        return None

    # Calculate the average line
    avg_x_start = sum_x_start // count
    avg_y_start = sum_y_start // count
    avg_x_end = sum_x_end // count
    avg_y_end = sum_y_end // count

    # Extend the average line to the edges of the box
    dx = avg_x_end - avg_x_start
    dy = avg_y_end - avg_y_start

    if dx != 0:
        t1 = (x1 - avg_x_start) / dx
        t2 = (x2 - avg_x_start) / dx
    else:
        t1 = float('-inf')
        t2 = float('inf')

    if dy != 0:
        t3 = (y1 - avg_y_start) / dy
        t4 = (y2 - avg_y_start) / dy
    else:
        t3 = float('-inf')
        t4 = float('inf')

    tmin = max(min(t1, t2), min(t3, t4))
    tmax = min(max(t1, t2), max(t3, t4))

    if tmax >= tmin and not (math.isnan(tmin) or math.isnan(tmax)):
        x_start = avg_x_start + tmin * dx
        y_start = avg_y_start + tmin * dy
        x_end = avg_x_start + tmax * dx
        y_end = avg_y_start + tmax * dy

        # Check for NaN values before converting to int
        if any(math.isnan(val) for val in [x_start, y_start, x_end, y_end]):
            return None

        return (int(x_start), int(y_start)), (int(x_end), int(y_end))
    else:
        return None


def mask_color(hsv_frame, color):
    if color == "red":
        lower = 330
        upper = 30
    elif color == "yellow":
        lower = 30
        upper = 90
    elif color == "green":
        lower = 90
        upper = 150
    elif color == "blue":
        lower = 210
        upper = 270
    else:
        lower = 0
        upper = 330
    
    mask_lower = hsv_frame[:,:,0].astype(np.uint16)*2 >= lower
    mask_upper = hsv_frame[:,:,0].astype(np.uint16)*2 < upper
    if color == "red":
        mask = mask_lower + mask_upper
    else:
        mask = mask_lower * mask_upper

    mask = mask.astype(np.uint8) * 255
    return mask

def find_brick_centers(mask, min_contour_area=75):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centers = []
    for contour in contours:
        # Only consider contours with an area larger than min_contour_area
        if cv2.contourArea(contour) >= min_contour_area:
            # Calculate the moments of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                # Calculate the center of the contour
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append((cX, cY))
    
    return centers

def finden_centers_of_color(hsv_frame, colors, box):
    x1, y1, x2, y2 = box
    all_centers = []
    for color in colors:
        mask = mask_color(hsv_frame[y1:y2, x1:x2], color)
        centers = find_brick_centers(mask)
        
        for center in centers:
            center = (center[0] + x1, center[1] + y1)
            all_centers.append(center)
            #cv2.circle(new_frame, center, 4, (255,0,255), -1)
    return all_centers

def stack_string(frame, centers, box, name_to_color):
    stack_str = ""
    line = centers_to_line(centers, box)
    if line:
        (x_start, y_start), (x_end, y_end) = line
        # cv2.line(new_frame, (x_start, y_start), (x_end, y_end), (255, 255, 255), 2)

    for i in range(0, 100):
        x = x_start + i * (x_end - x_start) // 100
        y = y_start + i * (y_end - y_start) // 100
        if x < 0 or x >= frame.shape[1] or y < 0 or y >= frame.shape[0]:
            continue
        hsv = frame[y][x]
        h,s,v = hsv
        color_name = get_color_name(h,s,v)
        if color_name in ["green", "blue","yellow","red"]:
            stack_str += color_name[0]
            cv2.circle(frame, (x, y), 1, name_to_color[color_name], 2)
    return stack_str

def draw_stack_box(frame, brick_coordinates, name_to_color):
    new_frame = frame.copy()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    stack_array_to_return = []
    stack_array = []

    (x_start, y_start), (x_end, y_end) = (0, 0), (0, 0)
    for (color, box) in brick_coordinates:
        # if color in bricks_in_frame:
        #     bricks_in_frame[color] = True

        x1, y1, x2, y2 = box
        
        stack_array = []
        all_centers = finden_centers_of_color(hsv_frame, ["green", "blue"], box)
   
        if len(all_centers) > 1:
            stack_str = ""
            line = centers_to_line(all_centers, box)
            if line:
                (x_start, y_start), (x_end, y_end) = line
                # cv2.line(new_frame, (x_start, y_start), (x_end, y_end), (255, 255, 255), 2)

            for i in range(0, 100):
                x = x_start + i * (x_end - x_start) // 100
                y = y_start + i * (y_end - y_start) // 100
                if x < 0 or x >= new_frame.shape[1] or y < 0 or y >= new_frame.shape[0]:
                    continue
                hsv = hsv_frame[y][x]
                h,s,v = hsv
                color_name = get_color_name(h,s,v)
                if color_name in ["green", "blue","yellow","red"]:
                    stack_str += color_name[0]
                    cv2.circle(new_frame, (x, y), 1, name_to_color[color_name], 2)
            
            stack_array = stack_str_to_array(stack_str)
            if len(stack_array) > len(stack_array_to_return):
                stack_array_to_return = stack_array

            if len(stack_array) > 1:
                #print(stack_str)
                #print(stack_array)

                for i, (color,_) in enumerate(stack_array):
                    #Satck 
                    cv2.putText(new_frame, f'{color}', (x2+10, y2 - 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 12)
                    cv2.putText(new_frame, f'{color}', (x2+10, y2 - 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, name_to_color[color], 2)
                    #Revese Stack
                    cv2.putText(new_frame, f'{color}', (x1-70, y1 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 12)
                    cv2.putText(new_frame, f'{color}', (x1-70, y1 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, name_to_color[color], 2)

       
        if len(stack_array) > 1:
            cv2.rectangle(new_frame, (x1, y1), (x2, y2), (0,0,0), 2)
            cv2.putText(new_frame, f'Stack', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        else:
            if color in name_to_color:
                cv2.rectangle(new_frame, (x1, y1), (x2, y2), name_to_color[color], 2)
                cv2.putText(new_frame, f'brick', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, name_to_color[color], 2)
            else:
                cv2.rectangle(new_frame, (x1, y1), (x2, y2), (0,0,0), 2)
                cv2.putText(new_frame, f'brick', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    return new_frame, stack_array_to_return
    
def find_stacks_and_bricks(frame):
    stacks_box, bricks_box = detect(frame, conf=0.4)
    
    res_bricks = []
    for box in bricks_box:
        [x, y, w, h] = box.xywh[0]
        x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x + w/2), int(y + h/2)
        
        w10 = int(w * 0.3)
        h10 = int(h * 0.3)
        x1_new = x1 + int(w * 0.35)
        y1_new = y1 + int(h * 0.35)
        x2_new = x1_new + w10
        y2_new = y1_new + h10

        # get average hsv value of the bounding box
        hsv = cv2.cvtColor(frame[y1_new:y2_new, x1_new:x2_new], cv2.COLOR_RGB2HSV)
        hsv = np.median(hsv,axis=(0,1))
        h,s,v = hsv
        detected_color_name = get_color_name(h,s,v)
        res_bricks.append((detected_color_name,(x1, y1, x2, y2)))

        cv2.circle(frame, (x + w/2, y + h/2), 1, (255,0,255), 2)
        




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