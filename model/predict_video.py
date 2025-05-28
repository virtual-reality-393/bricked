from brick import *
from tqdm import tqdm
import os



os.environ['YOLO_VERBOSE'] = 'False'

def predict_video(video_path : str, show_vid = True):


    detector = BrickDetector(multi_model=r"C:\Users\VirtualReality\Desktop\bricked\model\runs\detect\train63\weights\best.pt",is_video=False)
    video = cv2.VideoCapture(filename=video_path)

    out = cv2.VideoWriter('predicted.mp4', -1, 3, (1552//3,2064//3))

    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(frames)):
        frame_is_read, frame = video.read()
        
        multi_bboxes = detector.detect(frame,model_to_use=1)

        annotate_image(frame,multi_bboxes)

        if frame_is_read:
            frame = cv2.resize(frame,((1552//3,2064//3)))
            if show_vid:
                
                cv2.imshow("frame",frame)

                if cv2.waitKey(1) == ord("q"):
                    break
            out.write(frame)
        else:
            break

    out.release()

vid_name = "output"
predict_video(rf"C:\Users\VirtualReality\Desktop\bricked\model\unprocessed_data\raw_finished\{vid_name}.mp4",show_vid=True)