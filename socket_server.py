import asyncio
import websockets
import json
import cv2
from game_helper import GameHelper
from mss import mss
import numpy as np
from model.brick import *
import pygetwindow as gw
from PIL import Image
from io import BytesIO
import base64
import time
gameHelper = GameHelper()
# monitor = window.left+25, window.top+100, window.left + window.width-25, window.top + window.height-75

frame_num = 0

saved_frame_num = 0
accum = 0
FRAME_ACCUM = 2
sct = mss()
def handle_message(message):
    global frame_num,accum,saved_frame_num
    start = time.time()

    image = Image.open(BytesIO(base64.b64decode(message)))
    frame = np.array(image)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    

    # frame = cv2.resize(frame,(640,640))

    # gameHelper.process_frame(frame)

    # # # print("Handling Message",message)
    # # if message == "get_bricks":
    # #     
    # # elif message == "get_stacks":
    # #     return "[" + ', '.join([stack.toJSON() for stack in gameHelper.get_stacks()]) + "]"
    
    accum += (time.time()-start)

    if frame_num%FRAME_ACCUM == 0 and frame_num > 0: 

        print("FPS:",1/(accum/FRAME_ACCUM))
        print("Average Packet Time:",(accum/FRAME_ACCUM)*1000,"ms")
        frame_num = 0
        accum = 0
        saved_frame_num += 1
        frame_name = f"C:\\Users\\VirtualReality\\Desktop\\bricked\\headsetimgs\\{saved_frame_num}.jpg"

        print(cv2.imwrite(frame_name,frame))
        

    frame_num+=1

    return "[" + ', '.join([brick.toJSON() for brick in gameHelper.get_bricks()]) + "]"

# This function handles each client connection
async def echo(websocket):
    try:
        print(f"Client connected from {websocket.remote_address}")
        
        # Continuously listen for messages from the client
        async for message in websocket:            
            # Send a response back to the client
            await websocket.send(handle_message(message))

    except Exception as e:
        print(f"Error: {e}")
    finally:
        print(f"Client disconnected: {websocket.remote_address}")

# Start the server on port 8765
async def start_server():
    server = await websockets.serve(echo, "0.0.0.0", 8765,max_size=2**26)
    print("Server started on ws://0.0.0.0:8765")
    await server.wait_closed()

# Run the WebSocket server
asyncio.run(start_server())
