from brick import * 
from pathlib import Path
import glob


def annotate(in_path : str = "needs_annotation/", out_path : str = "processed_data/"):
    global image_points, image_label, mask, idx, save, image_paths,figure_bboxes
    if not Path.exists(Path(in_path)):
        raise IOError(f"in_path doesn't exist: {in_path}")
    if not Path.exists(Path(out_path)):
        raise IOError(f"out_path doesn't exist: {out_path}")
    
    image_paths = sorted(glob.glob(in_path + "*.npz"))
    idx = 0
    save = True
    figure_bboxes = {}
    while idx < len(image_paths):
        curr_idx = idx
        
            
        img_path = image_paths[curr_idx]

        data = list(np.load(img_path,allow_pickle=True).items())[0][1].item()
        org_img = data["org_img"]
        bboxes = detect(org_img,conf=0)
   

        


        for box in bboxes:
            # check if confidence is greater than 40 percent
                # get coordinates
                [x,y,w,h] = box.xywh[0]

                x1 = x
                x2 = x + w
                y1 = y
                y2 = y + h
                # convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # get the class
                cls = int(box.cls[0])

                # get the class name
                class_name = "brick"

                # draw the rectangle
                cv2.rectangle(org_img, (x1, y1), (x2, y2), (1,0,0), 2)

                # put the class name and confidence on the image
                cv2.putText(org_img, f'brick {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (1,0,0), 2)

        org_img = cv2.resize(org_img,np.transpose(np.array((org_img.shape[1],org_img.shape[0])))//data["factor"])

        # show the image
        cv2.imshow("frame", org_img)

        if cv2.waitKey(-1) == ord("q"):
            break
        if cv2.waitKey(-1) == ord("e"):
            idx+= 1
    cv2.destroyAllWindows()
        
annotate()