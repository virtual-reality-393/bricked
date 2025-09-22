import time

import cv2

from brick import *
from midi_player import MIDIPlayer

detector = BrickDetector()

cam = cv2.VideoCapture(2)

midi = MIDIPlayer()

time_test = time.time()

part = -1

play = True

class_names = ["red","green","blue","yellow"]

class_key = [50,55,60,65]

while True:

    ret, frame = cam.read()


    res = detector.detect(frame,model_to_use=1)
    offset = frame.shape[1]//4

    cv2.line(frame,(offset,0),(offset,frame.shape[0]),(0,0,255),2)

    cv2.line(frame, (offset*2, 0), (offset*2, frame.shape[0]), (0, 0, 255), 2)

    cv2.line(frame, (offset*3, 0), (offset*3, frame.shape[0]), (0, 0, 255), 2)

    if time.time()-time_test > 0.5:
        part = (part+1) % 4

        play = True

        time_test = time.time()




    if play:
        for detection in res:
            x, y, w, h = detection.xywh[0]
            cls = int(detection.cls[0].item())

            if cls >= 4: continue
            if offset * part < x < offset * (part+1):
                print(f"Playing note {class_names[cls]} with key, {class_key[cls]}")
                midi.play_note_async(class_key[cls],120,0.25)


        play = False

    cv2.imshow("video",frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



