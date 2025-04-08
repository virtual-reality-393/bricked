import asyncio
import websockets
import json
import cv2
from game_helper import GameHelper

gameHelper = GameHelper()


frame = cv2.imread(r"C:\Users\VirtualReality\Desktop\bricked\model\processed_data\0000156.jpg")
gameHelper.process_frame(frame)


print(len(gameHelper.get_stacks()))