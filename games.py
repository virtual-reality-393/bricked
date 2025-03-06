import cv2
import numpy as np
from mss import mss
from model.brick import *
import pygetwindow as gw
from model.assistant import VoiceAssistant
from game_helper import *

class StackingGame():
     
    def __init__(self):

        self.game_helper = GameHelper()
        self.game_visualizer = GameVisualizer()
        self.voice_assistant = VoiceAssistant()

        self.stack_to_build = []
        self.num_bricks_in_current_stack = 0
        self.bigest_stack =[]

        self.brick_used_for_stack = {}



    def play(self):
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

                self.game_helper.process_frame(frame)

                bricks = self.game_helper.get_bricks()
                stacks = self.game_helper.get_stacks()
                
                new_frame = frame.copy()

                current_stack = []

                self.game_visualizer.draw_stacks(new_frame, stacks)
                for brick in bricks:
                    if brick.in_stack != True:
                        self.game_visualizer.draw_brick(new_frame, brick) 
        
                if len(stacks) > 0:
                    order = self.game_helper.get_bigest_stack().get_order()
                    current_stack = []
                    for brick in order:
                        current_stack.append(brick.color)

                if state == "setup":
                    if len(bricks) >= 9 and len(stacks) == 0:
                        self.stack_to_build = self.game_helper.generate_stack_to_build({"red": 1, "green": 3, "blue": 2, "yellow": 3})
                        bigest_stack = []
                        state = "play"
                    else:
                        text = "Place all 9 bricks separated in frame"

                elif state == "play":
                    self.num_bricks_in_current_stack = len(current_stack)
                    if self.num_bricks_in_current_stack >= len(bigest_stack):
                        # print(f'stack to build: {self.stack_to_build}')
                        # print(f'current stack: {current_stack}')
                        # print(f'bigest stack: {bigest_stack}')
                        if self.num_bricks_in_current_stack < 2:
                            text = f"Start by stacking { self.stack_to_build[1]} on top of {self.stack_to_build[0]}"
                        elif current_stack == self.stack_to_build or current_stack == self.stack_to_build[::-1]:
                            text = "Stacking complete! Good job"
                            state = "setup"
                        elif current_stack == self.stack_to_build[:self.num_bricks_in_current_stack] or current_stack == self.stack_to_build[:self.num_bricks_in_current_stack][::-1]:
                            text = f'Next brick to add: { self.stack_to_build[self.num_bricks_in_current_stack]}'
                            bigest_stack =  self.stack_to_build[:self.num_bricks_in_current_stack]
                        # elif stack_array != stack_to_build[:self.num_bricks_in_current_stack] and stack_array != stack_to_build[:self.num_bricks_in_current_stack][::-1]:
                        #     text = "Wrong brick! Remove last brick" 
                        # else:
                        #     text = "debug text"

                self.game_visualizer.write_text(new_frame, text, (100, 170))

                if text != previos_text:
                    self.voice_assistant.play_message(text)
                previos_text = text

                # show the image
                cv2.imshow("frame", new_frame)

                # break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if cv2.waitKey(1) == ord("q"):
                    cv2.destroyAllWindows()
                    break

if __name__ == "__main__":
    stacking_game = StackingGame()
    stacking_game.play()