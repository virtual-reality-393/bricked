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
        self.current_stack_lifetime = 0
        self.brick_used_for_stack = {}



    def play(self):
        state = "setup"

        text = ""
        previos_text = ""

        prev_stack = None

        num_stacks_completed = 0

        self.voice_assistant.play_message("Welcome to the stacking game")

        with mss() as sct:
            while num_stacks_completed < 3:
                frame = self.game_helper.get_frame(sct)

                self.game_helper.process_frame(frame)

                bricks = self.game_helper.get_bricks()
                stacks = self.game_helper.get_stacks()
                
                new_frame = frame.copy()

                current_stack = []

                # Draw the stacks and bricks bounding boxes and brick centers
                self.game_visualizer.draw_stacks(new_frame, stacks)
                for brick in bricks:
                    if brick.in_stack != True:
                        self.game_visualizer.draw_brick(new_frame, brick) 
        
                # Get the order of the biggest stack in the frame
                if len(stacks) > 0:
                    order = self.game_helper.get_bigest_stack().get_order()
                    current_stack = []
                    for brick in order:
                        current_stack.append(brick.color)

                # Check if the current stack matches the previous stack
                if current_stack == prev_stack and current_stack != []:
                    self.current_stack_lifetime += 1 
                else:
                    self.current_stack_lifetime = 0 
                prev_stack = current_stack  

                # Game logic / game loop
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
                        if self.num_bricks_in_current_stack < 2:
                            text = f"Start by stacking { self.stack_to_build[1]} on top of {self.stack_to_build[0]}"
                        elif current_stack == self.stack_to_build or current_stack == self.stack_to_build[::-1]:
                            text = "Stacking complete! Good job"
                            num_stacks_completed += 1
                            state = "setup"
                        elif (current_stack == self.stack_to_build[:self.num_bricks_in_current_stack] or current_stack == self.stack_to_build[:self.num_bricks_in_current_stack][::-1]) and (len(current_stack) == len(self.stack_to_build[:self.num_bricks_in_current_stack]) and self.current_stack_lifetime > 10):
                            text = f'Next brick to add: { self.stack_to_build[self.num_bricks_in_current_stack]}'
                            bigest_stack =  self.stack_to_build[:self.num_bricks_in_current_stack]
                        elif (current_stack != self.stack_to_build[:self.num_bricks_in_current_stack] and current_stack != self.stack_to_build[:self.num_bricks_in_current_stack][::-1]) and self.current_stack_lifetime > 10:
                            if len(current_stack) >= len(self.stack_to_build[:self.num_bricks_in_current_stack]):
                                len_diff = len(current_stack) - len(bigest_stack)
                                if len_diff > 1:
                                    text = f"Wrong bricks! Remove last {len_diff} bricks"
                                else:
                                    text = "Wrong brick! Remove last brick"
                        # else:
                        #     text = "debug text"
                
                # Debug text
                self.game_visualizer.write_text(new_frame, f"Life time: {self.current_stack_lifetime}", (10, 130))
                self.game_visualizer.write_text(new_frame, text, (100, 170))

                # Play the voice message
                if text != previos_text:
                    self.voice_assistant.play_message(text)
                previos_text = text

                # show the image
                cv2.imshow("frame", new_frame)

                if cv2.waitKey(1) == ord("q"):
                    cv2.destroyAllWindows()
                    self.voice_assistant.play_message("KILL_PROCESS")
                    break
            
            # End of game
            self.voice_assistant.play_message("Congratulations! You have completed the game")
            cv2.destroyAllWindows()
            self.voice_assistant.play_message("KILL_PROCESS")

class MemoryGame():
        
    def __init__(self):

        self.game_helper = GameHelper()
        self.game_visualizer = GameVisualizer()
        self.voice_assistant = VoiceAssistant()

        self.stack_to_build = []

        self.brick_used_for_stack = {}
        self.current_stack_lifetime = 0



    def play(self):
        state = "setup"

        text = ""
        previos_text = ""

        num_bricks_in_stack = 3

        tell_stack = True

        prev_stack = None

        with mss() as sct:
            while True:
                frame = self.game_helper.get_frame(sct)

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
                
                # Check if the current stack matches the previous stack
                if current_stack == prev_stack and current_stack != []:
                    self.current_stack_lifetime += 1  # Increment lifetime if stack is the same
                else:
                    self.current_stack_lifetime = 0  # Reset lifetime if stack changes

                prev_stack = current_stack  # Update previous stac


                if state == "setup":
                    if len(bricks) >= 9 and len(stacks) == 0:
                        self.stack_to_build = self.generate_Memory_stack({"red": 1, "green": 3, "blue": 2, "yellow": 3},num_bricks_in_stack)
                        state = "play"
                    else:
                        text = "Place all 9 bricks separated in frame"

                elif state == "play":
                    if current_stack == self.stack_to_build or current_stack == self.stack_to_build[::-1]:
                        text = "Stacking complete! Good job"
                        if num_bricks_in_stack < 9:
                            num_bricks_in_stack += 1
                        state = "setup"
                    elif ((current_stack != self.stack_to_build and current_stack != self.stack_to_build[::-1]) and (len(current_stack) == len(self.stack_to_build) and self.current_stack_lifetime > 5)) or len(current_stack) > len(self.stack_to_build):
                        text = "Wrong stack! Try again"
                    elif tell_stack:
                        text = "Stack to build."
                        for brick in self.stack_to_build:
                            text += f" {brick}."
                        tell_stack = False
                    
                    if len(bricks) >= 9 and len(stacks) == 0:
                        tell_stack = True

                self.game_visualizer.write_text(new_frame, f"Life time: {self.current_stack_lifetime}", (10, 130))
                self.game_visualizer.write_text(new_frame, text, (10, 170))

                if text != previos_text:
                    self.voice_assistant.play_message(text)
                previos_text = text

                # show the image
                cv2.imshow("frame", new_frame)

                if cv2.waitKey(1) == ord("q"):
                    cv2.destroyAllWindows()
                    self.voice_assistant.play_message("KILL_PROCESS")
                    break
    
    def generate_Memory_stack(self,bricks_in_frame,num_bricks):
        #bricks = [color for color, count in bricks_in_frame.items() for _ in range(count)]
        bricks = ["red", "green","green","green", "blue","blue", "yellow","yellow","yellow"]
        stack_to_build = []

        random.shuffle(bricks)

        stack_to_build.append(bricks.pop(0))
    
        for brick in bricks:
            if stack_to_build[-1] != brick:
                stack_to_build.append(brick)
        
        if len(stack_to_build) >= num_bricks:
            stack_to_build = stack_to_build[:num_bricks]

        print(stack_to_build)
        return stack_to_build

#ToDo
class DoggoGame():

    def __init__(self):

        self.game_helper = GameHelper()
        self.game_visualizer = GameVisualizer()
        self.voice_assistant = VoiceAssistant()

        self.build_to_build = []

        self.brick_used_for_stack = {}
        self.current_stack_lifetime = 0



    def play(self):
        state = "setup"

        text = ""
        previos_text = ""

        prev_build = None

        with mss() as sct:
            while True:
                frame = self.game_helper.get_frame(sct)

                self.game_helper.process_frame(frame)

                bricks = self.game_helper.get_bricks()
                builds = self.game_helper.get_stacks()
                
                new_frame = frame.copy()

                current_build = []

                self.game_visualizer.draw_stacks(new_frame, builds)
                for brick in bricks:
                    if brick.in_stack != True:
                        self.game_visualizer.draw_brick(new_frame, brick) 
        
                if len(builds) > 0:
                    order = self.game_helper.get_bigest_stack()
                    current_stack = []
                    for brick in order:
                        current_stack.append(brick)
                
                # Check if the current build matches the previous build
                if current_build == prev_build and current_build != []:
                    self.current_stack_lifetime += 1  # Increment lifetime if build is the same
                else:
                    self.current_stack_lifetime = 0  # Reset lifetime if build changes

                prev_build = current_build  # Update previous build


                if state == "setup":
                    if len(bricks) >= 9 and len(builds) == 0: 
                        state = "play"
                    else:
                        text = "Place all 9 bricks separated in frame"

                elif state == "play":
                    if current_build != []:
                        # Check if the current build matches doggo
                        is_doggo = False

                        if is_doggo:
                            text = "Good job! You build the doggo"
                            state = "setup"
        
                self.game_visualizer.write_text(new_frame, f"Life time: {self.current_stack_lifetime}", (10, 130))
                self.game_visualizer.write_text(new_frame, text, (10, 170))

                # if text != previos_text:
                #     self.voice_assistant.play_message(text)
                # previos_text = text

                # show the image
                cv2.imshow("frame", new_frame)

                if cv2.waitKey(1) == ord("q"):
                    cv2.destroyAllWindows()
                    self.voice_assistant.play_message("KILL_PROCESS")
                    break

if __name__ == "__main__":
    stacking_game = MemoryGame()
    stacking_game.play()
