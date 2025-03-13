import cv2
import numpy as np
from model.brick import *
import random
import json


class Brick:
    
    def __init__(self, color, box):
        self.color = color
        self.box = box
        self.center = (box[0] + int((box[2] - box[0]) / 2), box[1] + int((box[3] - box[1]) / 2))
        self.in_stack = False

    def toJSON(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__,
            ensure_ascii=False)

class Stack:
    
    def __init__(self, bricks, box):
        self.bricks = bricks
        self.box = box
   
    def get_order(self):
        return self.__find_order_of_stack__(self.bricks)

    def __find_order_of_stack__(self, bricks):
        order = []
        res = []

        if len(bricks) == 0:
            return []

        if len(bricks) == 1:
            bricks[0].in_stack = True
            return [bricks[0]]
        
        if len(bricks) == 2:
            bricks[0].in_stack = True
            bricks[1].in_stack = True
            return [bricks[0], bricks[1]]

        for i, brick in enumerate(bricks):
            brick.in_stack = True
            center = brick.center
            dist1 = np.inf
            dist2 = np.inf
            id1 = 0
            id2 = 0
            vec1 = None
            vec2 = None
            for j,brick2 in enumerate(bricks):
                if brick != brick2:
                    center2 = brick2.center
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
            res.append(bricks[i])

        return res
    
    def toJSON(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__,
            ensure_ascii=False)




class GameHelper:

    def __init__(self):
        self.brickDetector = BrickDetector()

        self.task_completed = False
        
        self.bricks_in_frame = {"red": 0, "green": 0, "blue": 0, "yellow": 0}
        self.name_to_color = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0), "yellow": (0, 255, 255), "magenta": (255, 0, 255)}
        
        self.bricks = []
        self.stacks = []

        self.bigest_stack = None


    def process_frame(self, frame):
        stacks, bricks = self.__find_stacks_and_bricks__(frame, self.brickDetector)

        res_stacks = []
        for stack in stacks:
            if len(stack.bricks) > 1:
                res_stacks.append(stack)
                # if self.bigest_stack == None:
                #     self.bigest_stack = stack

                # if len(stack.bricks) > len(self.bigest_stack.bricks):
                #     self.bigest_stack = res_stacks[-1]

        res_bricks = []
        self.bricks_in_frame = {"red": 0, "green": 0, "blue": 0, "yellow": 0}    
        for brick in bricks:
            self.bricks_in_frame[brick.color] += 1
            res_bricks.append(brick)


        self.bricks = res_bricks
        self.stacks = res_stacks

    def generate_stack_to_build(self,bricks_in_frame, default=False):
        #bricks = [color for color, count in bricks_in_frame.items() for _ in range(count)]
        bricks = ["red", "green","green","green", "blue","blue", "yellow","yellow","yellow"]
        stack_to_build = []

        random.shuffle(bricks)

        stack_to_build.append(bricks.pop(0))
    
        for brick in bricks:
            if stack_to_build[-1] != brick:
                stack_to_build.append(brick)
    
        if default:
            stack_to_build = ["blue", "green", "yellow", "red", "green", "yellow", "blue", "yellow", "green"]

        print(stack_to_build)
        return stack_to_build

    def get_stacks(self):
        return self.stacks.copy()
    
    def get_bricks(self):
        return self.bricks.copy()

    def get_bricks_in_frame(self):
        return self.bricks_in_frame.copy()

    def get_num_bricks_in_frame(self):
        return sum([self.bricks_in_frame[color] for color in self.bricks_in_frame])

    def get_bigest_stack(self):
        if self.stacks == []:
            return Stack([],(0,0,0,0))
        self.bigest_stack = None
        for stack in self.stacks:
            if self.bigest_stack == None:
                self.bigest_stack = stack

            if len(stack.bricks) > len(self.bigest_stack.bricks):
                self.bigest_stack = stack

        if self.bigest_stack == None:
            return Stack([],(0,0,0,0))
        return self.bigest_stack

    def get_next_brick(self):
        return self.stacks_to_build[len(self.bigest_stack)]

    def __find_stacks_and_bricks__(self,frame, brickdetector):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        stacks_box, bricks_box = brickdetector.detect(frame, conf=0.4)

        res_bricks = []
        res_stacks = []

        for box in bricks_box:
            [x, y, w, h] = box.xywh[0]
            x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x + w/2), int(y + h/2)

            x1_new = max(0, int(x - w * 0.05))
            y1_new = max(0, int(y - h * 0.05))
            x2_new = min(frame.shape[1], int(x + w * 0.05))
            y2_new = min(frame.shape[0], int(y + h * 0.05))

            # Ensure the region is valid
            if x1_new >= x2_new or y1_new >= y2_new:
                continue

            # get average hsv value of the bounding box
            hsv = hsv_frame[y1_new:y2_new, x1_new:x2_new]
            hsv = np.median(hsv, axis=(0,1))
            h, s, v = hsv
            detected_color_name = self.__get_color_name__(h, s, v)

            if detected_color_name in ["green", "blue","yellow","red"]:
                res_bricks.append(Brick(detected_color_name, (x1, y1, x2, y2)))

        for stack_box in stacks_box:
            [x, y, w, h] = stack_box.xywh[0]
            x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x + w/2), int(y + h/2)

            # Find centers in bricks_box that are inside the stack_box
            bricks_in_stack = []
            for brick in res_bricks:
                center_x, center_y = brick.center
                if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                    if len(bricks_in_stack) > 0 and bricks_in_stack[-1].color != brick.color:
                        bricks_in_stack.append(brick)

            res_stacks.append(Stack(bricks_in_stack, (x1, y1, x2, y2)))

        return res_stacks, res_bricks  
    
    def __get_color_name__(self,h,s,v):
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

    
class GameVisualizer():
    
    def __init__(self):
        self.name_to_color = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0), "yellow": (0, 255, 255), "magenta": (255, 0, 255)}
    
    def write_text(self, frame, text, position, color=(255,255,255)):
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 12)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def draw_brick(self, frame, brick):
        x1, y1, x2, y2 = brick.box
        cv2.circle(frame, brick.center, 7, (0,0,0), -1)
        cv2.circle(frame, brick.center, 4, (255,0,255), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.name_to_color[brick.color], 2)
        self.write_text(frame, f'{brick.color} brick', (x1, y1 - 10), self.name_to_color[brick.color])

    def draw_stacks(self, frame, stacks):
        for j, stack in enumerate(stacks):
            order = stack.get_order()
            x1, y1, x2, y2 = stack.box
            if len(order) > 1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                self.write_text(frame, f'Stack {j}', (x1, y1 - 10))
                for i, brick in enumerate(order):
                    color = brick.color
                    #Satck 
                    self.write_text(frame, f'{color}', (x2+10, y2 - 22 * i), self.name_to_color[color])
                    #Revese Stack
                    self.write_text(frame, f'{color}', (x1-70, y1 + 22 * i), self.name_to_color[color])
                    cv2.circle(frame, brick.center, 7, (0,0,0), -1)
                    cv2.circle(frame, brick.center, 4, (255,0,255), -1)