import cv2
import numpy as np
from mss import mss
from model.brick import *
from util import *
import pygetwindow as gw
#from model.assistant import VoiceAssistant


if __name__ == "__main__":
    #assistant = VoiceAssistant(voice = "model/models/joe")
    brik_centers = {}
    target = None
    brik_to_move = None
    task_completed = False
    dist_to_target = {}
    briks_in_frame = {"red": False, "green": False, "blue": False, "yellow": False}
    name_to_color = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0), "yellow": (0, 255, 255)}
    
    text = ""
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
            
            num_briks = sum([briks_in_frame[color] for color in briks_in_frame])
            if num_briks > 1:
                if target == None:
                    target = np.random.choice([color for color in briks_in_frame if briks_in_frame[color]])
                if brik_to_move == None:
                    brik_to_move = np.random.choice([color for color in briks_in_frame if color != target and briks_in_frame[color]])
                text = f"Move {brik_to_move} to {target}"
                #assistant.play_message(text)
            else:
                target = None
                brik_to_move = None
                text = "Not enough bricks in frame"
                #assistant.play_message(text)

            if target != None:
                if not briks_in_frame[target]:
                    text = f'{target} brick not in frame'
                    #target = None
            if brik_to_move != None:
                if not briks_in_frame[brik_to_move]:
                    text = f'{brik_to_move} brick not in frame'
                    #brik_to_move = None

            lineColor = (255,255,255)
            if target != None and brik_to_move != None:
                if briks_in_frame[brik_to_move] and briks_in_frame[target]:
                    dist_to_target[brik_to_move] = np.linalg.norm(np.array(brik_centers[target]) - np.array(brik_centers[brik_to_move]))
                    
                    if dist_to_target[brik_to_move] < 100 and not task_completed:
                        task_completed = True
                        lineColor = (0,255,0)
                        text = "Good Job"
                        #assistant.play_message(text)

                    elif dist_to_target[brik_to_move] < 150 and task_completed:
                        lineColor = (0,0,255)
                        text = f"Move {brik_to_move} and {target} further apart"
                        #assistant.play_message(text)
                    
                    elif (dist_to_target[brik_to_move] > 150 and task_completed):
                        task_completed = False
                        target = np.random.choice([color for color in briks_in_frame if briks_in_frame[color]])
                        brik_to_move = np.random.choice([color for color in briks_in_frame if color != target and briks_in_frame[color]])
                        text = f"Move {brik_to_move} to {target}"
                        #assistant.play_message(text)
                if briks_in_frame[brik_to_move] and briks_in_frame[target]:
                    cv2.line(frame, brik_centers[brik_to_move], brik_centers[target], lineColor, 2)

            cv2.putText(frame, text, (120, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, lineColor, 2)
            cv2.putText(frame, f'Num briks {num_briks}', (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f'Target: {target}', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f'Brik to move: {brik_to_move}', (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)



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