from brick import *
import glob
import math
import matplotlib.pyplot as plt
import random
import os
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import glob
import math
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import shutil
from collections import namedtuple

def __onkeypress__(event):
    global save,idx, save_override
    if event.key == "left":
        idx = max(idx-1,0)
        plt.close()
    if event.key == "right":
        idx = min(idx+1,len(image_paths))
        plt.close()
    if event.key == " ":
        idx = min(idx+1,len(image_paths))
        plt.close()
    if event.key == "up":
        save_override = True
        idx = min(idx+1,len(image_paths))
        plt.close()

    if event.key == "q":
        exit()

def __pre_annotate__(image,factor):
    bboxes = detect(image,conf = 0.4)
    Event = namedtuple("Event",["button","xdata","ydata"])
    for box in bboxes:
        [x,y,w,h] = box.xywh[0]
        event = Event(1,(x+w//2)//factor,(y+h//2)//factor)
        __onclick__(event)



    


def __onclick__(event): 
    global curr_idx, figure_bboxes

    image_label.append(1 if event.button == 1 else 0)
    image_points.append((int(event.xdata),int(event.ydata)))
    fig = plt.gcf()
    i = int(event.xdata)
    j = int(event.ydata)

    i = min(mask.shape[1]-1,i)
    j = min(mask.shape[0]-1,j)

    val = int(mask[j,i])

    if val == 0:
        return

    num,_,flood_mask,rect = cv2.floodFill(mask,np.zeros((mask.shape[0]+2,mask.shape[1]+2),np.uint8),seedPoint=(i,j),newVal=val)

    ax = fig.get_axes()[0]
    
    if event.button == 1:        
        x,y,w,h = rect
        # rectangle = plt.Rectangle((i,j),10,10, linewidth=4, color=(1,0,0,0.4), fill=True) # Debug
        # rect_patch = ax.add_patch(rectangle)
        if rect not in figure_bboxes[curr_idx]:
            rectangle = plt.Rectangle((x,y),w,h, linewidth=4, color=(1,0,0,0.4), fill=True)
            


            rect_patch = ax.add_patch(rectangle)


            figure_bboxes[curr_idx][rect] = (rect_patch,rectangle)

    if event.button == 3:
    
        if rect in figure_bboxes[curr_idx]:
            figure_bboxes[curr_idx][rect][0].remove()
            figure_bboxes[curr_idx].pop(rect)
        

   
    fig.canvas.draw()

def process_video(in_path : str = "unprocessed_data/", out_path : str = "unprocessed_data/"):

    vid_paths = glob.glob(in_path + "*.mp4")

    if len(vid_paths) ==  0:
        print("No videos to process")
        return
    
    img_idx = 0
    for vid_path in vid_paths:
        print(f"Processing {vid_path}")
        video_capture = cv2.VideoCapture(vid_path)
        saved_frame_name = 0
        while video_capture.isOpened():
            frame_is_read, frame = video_capture.read()

            if frame_is_read:
                if saved_frame_name % 30 == 0:
                    cv2.imwrite(f"{out_path}{str(img_idx)}.jpg", frame)
                    img_idx+=1
                saved_frame_name += 1
            else:
                break
        print(f"Saved {img_idx} images from {vid_path}")
        video_capture.release()
        os.rename(vid_path, in_path + "raw_finished/" + Path(vid_path).name)
    

    print(f"Saved {img_idx} images in total")
    
    

def __generate_visual_segmentation__(anns, file_name,img,org,scalefactor, borders=True):
    if len(anns) == 0:
        return
    
    image = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

    image = image.astype("float64")/255
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    segment_image = image
    segment_mask = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]),dtype=np.uint8)
    for i,ann in enumerate(sorted_anns):
        m = ann['segmentation']

        segment_mask[m] = i
        color_mask = np.concatenate([np.array([0,0.5,0.5]), [1]])
        segment_image[m] = segment_image[m]*0.5 + color_mask*0.5
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(segment_image, contours, -1, (0, 0, 1, 0.4), thickness=1)

    data = {"org_img": org,"segment_img" : segment_image, "mask" : segment_mask,"factor":scalefactor}
    np.savez_compressed(file_name,data)
    

def annotate(in_path : str = "needs_annotation/", out_path : str = "processed_data/",start_index : int = -1):
    global image_points, image_label, mask, curr_idx,idx, save, image_paths,figure_bboxes, save_override
    if not Path.exists(Path(in_path)):
        raise IOError(f"in_path doesn't exist: {in_path}")
    if not Path.exists(Path(out_path)):
        raise IOError(f"out_path doesn't exist: {out_path}")
    
    image_paths = sorted(glob.glob(in_path + "*.npz"))
    idx = 0
    save = True
    figure_bboxes = {}
    save_override = False

    if start_index == -1:
        start_index = int(read_file("annotate_point.txt")[0])


    if len(image_paths) <= start_index:
        print("No images to annotate")
        return
    
    while idx < len(image_paths):
        curr_idx = idx + start_index

            
        img_path = image_paths[curr_idx]
        

        data = list(np.load(img_path,allow_pickle=True).items())[0][1].item()


        image = data["segment_img"]
        org_img = data["org_img"]
        mask = data["mask"]
        factor = data["factor"]
        image[mask == 0] *= 0.3
        image[mask == 0][3] = 1 
        image_points = []
        image_label = []

        

        ax = plt.imshow(image)
        plt.axis('off')
        fig = ax.get_figure()

        fig.set_size_inches(15,15)

        fig.canvas.manager.window.wm_geometry("+%d+%d" % (0, -100))

        if curr_idx not in figure_bboxes:
            figure_bboxes[curr_idx] = {}
            __pre_annotate__(org_img,factor)

        for rect in figure_bboxes[curr_idx]:
            figure_bboxes[curr_idx][rect][0].remove()
            fig.get_axes()[0].add_patch(figure_bboxes[curr_idx][rect][1])

        

        cid = fig.canvas.mpl_connect('button_press_event', __onclick__) 
        cid = fig.canvas.mpl_connect('key_press_event', __onkeypress__) 
        plt.show() 

        file_name = str(curr_idx)

        file_name = file_name.zfill(7)

        labels_string = ""

        for x,y,w,h in figure_bboxes[curr_idx]:
            labels_string += f"0 {x/image.shape[1]} {y/image.shape[0]} {w/image.shape[1]} {h/image.shape[0]}\n"

        if len(figure_bboxes[curr_idx]) > 0 or save_override:
            if save:
                labels_string = labels_string.rstrip("\n")
        
                plt.imsave(f"processed_data/{file_name}.jpg",org_img)

                with open(f"processed_data/{file_name}.txt","w") as text_file:
                    text_file.write(labels_string)
                save_override = False
        save = True
    
    start_index = write_file("annotate_point.txt",str(curr_idx))


def process_raw_images(in_path : str = "unprocessed_data/", out_path : str = "needs_annotation/"):
    MODEL = "sam2.1_hiera_base_plus"

    if not Path.exists(Path(in_path)):
        raise IOError(f"in_path doesn't exist: {in_path}")

    if not Path.exists(Path(out_path)):
        Path.mkdir(out_path)
        print(f"out_path doesn't exist: {out_path}, creating it instead")
    

    image_paths = glob.glob(in_path + "*.jpg")

    checkpoint = f"./checkpoints/{MODEL}.pt"
    model_cfg = f"configs/sam2.1/{MODEL}.yaml"
    sam2 = build_sam2(model_cfg, checkpoint, device="cuda", apply_postprocessing=False)

    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    curr_highest = max([int(Path(path).name.split(".")[0]) for path in glob.glob(out_path+"*.npz")]) + 1

    for i,img_path in enumerate(image_paths):

        print(f"Processing Image {img_path}")

        image = load_image(img_path)

        scale_factor = max(int(math.floor(min(image.shape[:2]) / 512.0)),1)
        image_shape = (int(image.shape[1]/scale_factor), int(image.shape[0]/scale_factor))
        scaled_image = cv2.resize(image, image_shape)

        masks = mask_generator.generate(scaled_image)

        file_name = str(i + curr_highest)

        file_name = file_name.zfill(7)

        __generate_visual_segmentation__(masks,out_path + file_name,scaled_image,image,scale_factor)

        os.remove(img_path)

def create_splits(in_path : str = "processed_data/", out_path : str = "datasets/brick/", splits : list = [("train",0.8),("val",0.2)]):
    random.seed(424354)

    if not Path.exists(Path(in_path)):
        raise IOError(f"in_path doesn't exist: {in_path}")

    if not Path.exists(Path(out_path)):
        Path.mkdir(out_path)
        print(f"out_path doesn't exist: {out_path}, creating it instead")
    

    image_paths = glob.glob(in_path + "*.jpg")
    label_paths = glob.glob(in_path + "*.txt")

    image_paths = sorted(image_paths)
    label_paths = sorted(label_paths)

    splits = sorted(splits,key=lambda x: x[1])


    
    split_count = {name:0 for name,val in splits}

    for i in range(len(image_paths)):
        rnd = random.random()
        split_val = 0
        img_path = Path(image_paths[i])
        label_path = Path(label_paths[i])

        for name,val in splits:
            split_val += val
            if rnd < split_val:
                split_count[name] += 1
                os.makedirs(os.path.dirname(f"{out_path}images/{name}/{img_path.name}"), exist_ok=True)
                shutil.copy(img_path,f"{out_path}images/{name}/{img_path.name}")
                os.makedirs(os.path.dirname(f"{out_path}labels/{name}/{label_path.name}"), exist_ok=True)
                shutil.copy(label_path,f"{out_path}labels/{name}/{label_path.name}")
                break

    print(split_count)


        
if __name__ == "__main__":
    process_video()
    process_raw_images()
    annotate()
    create_splits()








