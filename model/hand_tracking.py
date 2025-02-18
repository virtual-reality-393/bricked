import cv2
import mediapipe as mp
from brick import *
import pygetwindow as gw
from mss import mss

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2,
    min_detection_confidence=0.2)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

with mss() as sct:
    while True:

        
        # monitor = sct.monitors[2]
        window = gw.getWindowsWithTitle("Pixel 6 Pro")[0]
        monitor = window.left+15, window.top+250, window.left + window.width-15, window.top + window.height-30
        
        sct_img = sct.grab(monitor)

        frame = np.array(sct_img)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        results = hands.process(frame)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(id, cx, cy)
                    if id == 4 :
                        cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                    if id == 8 :
                        cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                    if id == 12 :
                        cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                    if id == 16 :
                        cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                    if id == 20 :
                        cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)


        # show the image
        cv2.imshow("frame", frame)

        # break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break
