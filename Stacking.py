import cv2
import numpy as np
from mss import mss
from model.brick import *
from util import *
import pygetwindow as gw
from model.assistant import VoiceAssistant

def putText(frame, text, x, y, color=(0, 0, 0), size=0.7, use_voice=False, assistant=None):

    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (0,0,0), 12)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)

    if use_voice and assistant:
        assistant.play_message(text)

    return frame

if __name__ == "__main__":
    use_voice = True
    if use_voice:
        assistant = VoiceAssistant()
        pass

    brickDetector = BrickDetector()
    brick_centers = {}
    target = None
    brick_to_move = None
    task_completed = False
    dist_to_target = {}
    bricks_in_frame = {"red": 0, "green": 0, "blue": 0, "yellow": 0}
    name_to_color = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0), "yellow": (0, 255, 255), "black": (0, 0, 0), "white": (255, 255, 255), "cyan": (255, 255, 0), "magenta": (255, 0, 255), "unknown": (255, 255, 255)}

    stack_to_build = []
    num_bricks_in_current_stack = 0
    bigest_stack =[]

    brick_used_for_stack = {}

    state = "setup"

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

            stacks, bricks, new_frame = find_stacks_and_bricks(frame, brickDetector)
            
            for brick in bricks:
                x1, y1, x2, y2 = brick[1]
                cv2.circle(new_frame, (x1 + int((x2 - x1) / 2), y1 + int((y2 - y1) / 2)), 7, (0,0,0), -1)
                cv2.circle(new_frame, (x1 + int((x2 - x1) / 2), y1 + int((y2 - y1) / 2)), 4, (255,0,255), -1)
                # cv2.rectangle(new_frame, (x1, y1), (x2, y2), name_to_color[brick[0]], 2)
                # cv2.putText(new_frame, f'{brick[0]} brick', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, name_to_color[brick[0]], 2)

            stack_array = []
            for j, stack in enumerate(stacks):
                order = find_order_of_stack(stack)
                x1, y1, x2, y2 = stack[1]
                if len(order) > len(stack_array):
                    stack_array = order

                if len(order) > 1:
                    cv2.rectangle(new_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                    cv2.putText(new_frame, f'Stack {j}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 12)
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

            if state == "setup":
                bricks_in_frame = {"red": 0, "green": 0, "blue": 0, "yellow": 0}
                for (color, box) in bricks:
                    if color in bricks_in_frame:
                        bricks_in_frame[color] += 1
                        cv2.putText(new_frame, f'{color} bricks: {bricks_in_frame[color]}', (10, 30 + 30 * list(bricks_in_frame.keys()).index(color)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 12)
                        cv2.putText(new_frame, f'{color} bricks: {bricks_in_frame[color]}', (10, 30 + 30 * list(bricks_in_frame.keys()).index(color)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, name_to_color[color], 2)
                if sum(bricks_in_frame.values()) >= 9 and len(stack_array) == 1:
                    brick_used_for_stack = bricks_in_frame
                    stack_to_build = generate_stack_to_build({"red": 1, "green": 3, "blue": 2, "yellow": 3})
                    bigest_stack = []
                    state = "play"
                else:
                    text = "Place all 9 bricks separated in frame"

            elif state == "play":

                # for color in brick_used_for_stack:
                #     cv2.putText(new_frame, f'{color} bricks: {brick_used_for_stack[color]}', (10, 30 + 30 * list(brick_used_for_stack.keys()).index(color)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 12)
                #     cv2.putText(new_frame, f'{color} bricks: {brick_used_for_stack[color]}', (10, 30 + 30 * list(brick_used_for_stack.keys()).index(color)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, name_to_color[color], 2)

                num_bricks_in_current_stack = len(stack_array)
                if num_bricks_in_current_stack >= len(bigest_stack):
                    if num_bricks_in_current_stack < 2:
                        text = f"Start by stacking {stack_to_build[1]} on top of {stack_to_build[0]}"
                    elif stack_array == stack_to_build or stack_array == stack_to_build[::-1]:
                        text = "Stacking complete! Good job"
                        state = "setup"
                    elif stack_array == stack_to_build[:num_bricks_in_current_stack] or stack_array == stack_to_build[:num_bricks_in_current_stack][::-1]:
                        text = f'Next brick to add: {stack_to_build[num_bricks_in_current_stack]}'
                        bigest_stack = stack_to_build[:num_bricks_in_current_stack]
                    # elif stack_array != stack_to_build[:num_bricks_in_current_stack] and stack_array != stack_to_build[:num_bricks_in_current_stack][::-1]:
                    #     text = "Wrong brick! Remove last brick" 
                    # else:
                    #     text = "debug text"

            if text != previos_text:
                new_frame = putText(new_frame, text, 110, 150, (255,255,255), 0.7, True, assistant)
            else:
                new_frame = putText(new_frame, text, 110, 150, (255,255,255), 0.7, False)
            
            previos_text = text

            # show the image
            cv2.imshow("frame", new_frame)

            # break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                break