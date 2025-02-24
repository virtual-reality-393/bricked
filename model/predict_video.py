from brick import *

def predict_video(video_path : str):
    video = cv2.VideoCapture(filename=video_path)

    while video.isOpened():
        frame_is_read, frame = video.read()

        frame = cv2.resize(frame,(frame.shape[1],frame.shape[0]))


        bboxes = detect(frame,is_video=True)

        annotate_image(frame,bboxes)

        if frame_is_read:
            cv2.imshow("frame", frame)


            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                break
        else:
            break

vid_name = "20250224_094408_c6d9126d"
predict_video(rf"C:\Users\VirtualReality\Desktop\bricked\model\unprocessed_data\raw_finished\{vid_name}.mp4")