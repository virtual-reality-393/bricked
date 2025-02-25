import cv2
import numpy as np
from mss import mss
from model.brick import *
from util import *
import pygetwindow as gw
#from model.assistant import VoiceAssistant


def draw_box_coordinat(frame, brick_coordinates):
    new_frame = frame.copy()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brick_color = ["green", "blue"]
    for (color, box) in brick_coordinates:
        if color in bricks_in_frame:
            bricks_in_frame[color] = True

        x1, y1, x2, y2 = box

        drawColor =(0,0,0)
        if color in name_to_color:
            drawColor = name_to_color[color]

        drawColor =(0,0,0)
        
        cv2.rectangle(new_frame, (x1, y1), (x2, y2), drawColor, 2)
        #cv2.putText(new_frame, f'brick', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (drawColor), 2)

        all_centers = []
        for c in brick_color:
            mask = mask_color(hsv_frame[y1:y2, x1:x2], c)
            centers = find_brick_centers(mask)
            
            for center in centers:
                center = (center[0] + x1, center[1] + y1)
                all_centers.append(center)
                #cv2.circle(new_frame, center, 4, (255,0,255), -1)
   
        if len(all_centers) > 1:
            stack_str = ""
            line = centers_to_line(all_centers, box)
            if line:
                (x_start, y_start), (x_end, y_end) = line
                cv2.line(new_frame, (x_start, y_start), (x_end, y_end), (255, 255, 255), 2)

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
            # print(stack_str)
            # print(stack_array)

            for i, (color,_) in enumerate(stack_array):
                cv2.putText(new_frame, f'{color}', (x2+10, y2 - 30 - 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 12)
                cv2.putText(new_frame, f'{color}', (x2+10, y2 - 30 - 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, name_to_color[color], 2)

    return new_frame

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

    #calculate the avager char count
    stack_avage = 0
    for _, count in stack_array:
        stack_avage += count
    stack_avage = stack_avage / len(stack_array)
    stack_array = [(color, count) for color, count in stack_array if count > stack_avage*0.8]


    #stack_array = stack_array[1:-1]
    return stack_array


def centers_to_line(centers, box):
    # Calculate the parametric form of the line passing through the centers
    x1, y1, x2, y2 = box
    (x1_center, y1_center), (x2_center, y2_center) = centers[0], centers[-1]
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

    if tmax >= tmin:
        x_start = int(x1_center + tmin * dx)
        y_start = int(y1_center + tmin * dy)
        x_end = int(x1_center + tmax * dx)
        y_end = int(y1_center + tmax * dy)

        return (x_start, y_start), (x_end, y_end)
    else:
        return None

def mask_color(hsv_frame, color):
    if color == "red":
        lower = 30
        upper = 330
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
    mask = mask_lower * mask_upper
    mask = mask.astype(np.uint8) * 255 

    return mask



def find_brick_centers(mask):
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centers = []
    for contour in contours:
        # Calculate the moments of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            # Calculate the center of the contour
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
    
    return centers

if __name__ == "__main__":
    use_voice = False
    if use_voice:
        #assistant = VoiceAssistant(voice = "model/models/joe")
        pass

    brick_centers = {}
    target = None
    brick_to_move = None
    task_completed = False
    dist_to_target = {}
    bricks_in_frame = {"red": False, "green": False, "blue": False, "yellow": False}
    name_to_color = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0), "yellow": (0, 255, 255)}
    
    text = ""
    previos_text = ""
    with mss() as sct:
        while True:
            # monitor = sct.monitors[2]
            window = gw.getWindowsWithTitle("Pixel 6 Pro")[0]
            monitor = window.left+25, window.top+100, window.left + window.width-25, window.top + window.height-75
            # monitor = window.left, window.top, window.left + window.width, window.top + window.height
            
            sct_img = sct.grab(monitor)

            frame = np.array(sct_img)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            brick_coordinates = find_brick_and_color(frame, is_video=True)

            new_frame = draw_box_coordinat(frame, brick_coordinates)



            # show the image
            cv2.imshow("frame", new_frame)

            # break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                break