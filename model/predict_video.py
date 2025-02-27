from brick import *
from tqdm import tqdm
import os

os.environ['YOLO_VERBOSE'] = 'False'

def predict_video(video_path : str):


    detector = BrickDetector()
    video = cv2.VideoCapture(filename=video_path)

    out = cv2.VideoWriter('predicted.mp4', -1, 25, (1552//3,2064//3))

    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(frames)):
        frame_is_read, frame = video.read()
        
        single_bboxes,multi_bboxes = detector.detect(frame,model_to_use=2)

        annotate_image(frame,single_bboxes)
        annotate_image(frame,multi_bboxes)

        if frame_is_read:
            frame = cv2.resize(frame,(frame.shape[1]//3,frame.shape[0]//3))
            cv2.imshow("frame",frame)

            cv2.waitKey(1)
            out.write(frame)
        else:
            break

vid_name = "test_video"
predict_video(rf"C:\Users\VirtualReality\Desktop\bricked\model\unprocessed_data\raw_finished\{vid_name}.mp4")