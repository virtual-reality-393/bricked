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
from tqdm import tqdm


class Annotator:
    def __init__(self, seg_model : str = "sam2.1_hiera_base_plus",model_to_use = 1,unproc = "unprocessed_data/", need_anno = "needs_annotation/", proc = "processed_data/"):
        if not Path.exists(Path(unproc)):
            raise IOError(f"Path to unprocessed data doesn't exist: {unproc}")

        if not Path.exists(Path(need_anno)):
            raise IOError(f"Path to annotation needed data doesn't exist: {need_anno}")
        
        if not Path.exists(Path(proc)):
            raise IOError(f"Path to processed data doesn't exist: {proc}")
        
        self.unproc = unproc
        self.need_anno = need_anno
        self.seg_model = seg_model
        self.model_to_use = model_to_use
        self.detector_model = BrickDetector(is_video=False,multi_model="models/run64_stacks.pt")

        self.label_idx_to_name = {0:"red",1:"green",2:"blue",3:"yellow",4:"big penguin",5:"small penguin",6:"lion",7:"sheep",8:"pig",9:"human"}

        self.label_idx_to_color = {0:(1,0,0,1),1:(0,1,0,1),2:(0,0,1,1),3:(1,1,0,1),4:(0.4,0.4,1,1),5:(0.4,0.4,0.4,1),6:(1,0.5,0,1),7:(1,1,1,1),8:(1,0.5,0.5,1),9:(1.0,0.2,0.8,1)}

        self.__mask_generator__ = None

        self.annotation_class = 0

        self.rects = {}
        self.extra_anno = {}

        self.curr_image = {}

        self.bbox_active = False
        self.select_bbox = []

    def __get_mask_generator__(self):
        if self.__mask_generator__ == None:
            checkpoint = f"./models/{self.seg_model}.pt"
            model_cfg = f"configs/sam2.1/{self.seg_model}.yaml"
            
            sam2 = build_sam2(model_cfg, checkpoint, device="cuda", apply_postprocessing=False)

            mask_generator = SAM2AutomaticMaskGenerator(sam2)
            self.__mask_generator__ = mask_generator

        return self.__mask_generator__
    
    def __on_click__(self,event,x,y,flags,params):
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            i = int(x)
            j = int(y)

            mask = self.curr_image["mask"]

            i = min(mask.shape[1]-1,i)
            j = min(mask.shape[0]-1,j)

            val = int(mask[j,i])

            if val == 0:
                return

            num,_,flood_mask,rect = cv2.floodFill(mask,np.zeros((mask.shape[0]+2,mask.shape[1]+2),np.uint8),seedPoint=(i,j),newVal=val)

            if self.curr_idx not in self.rects:
                self.rects[self.curr_idx] = []

            if event == cv2.EVENT_LBUTTONDOWN:
                if all([(anno_class,*rect) not in self.rects[self.curr_idx] for anno_class in range(0,10)]):
                    self.rects[self.curr_idx].append((self.annotation_class,*rect))

            if event == cv2.EVENT_RBUTTONDOWN:
                for anno_class in range(0,10):
                    if (anno_class,*rect) in self.rects[self.curr_idx]:
                        self.rects[self.curr_idx].remove((anno_class,*rect))

            if params[0]:
                self.__update_image__()
            else:
                if self.curr_idx not in self.extra_anno:
                    self.extra_anno[self.curr_idx] = []

                self.extra_anno[self.curr_idx].append((x,y))

        if event == cv2.EVENT_MBUTTONDOWN:
            self.bbox_active = True
            self.bbox_start = (x,y)

        if event == cv2.EVENT_MBUTTONUP:
            self.bbox_active = False

            (x2,y2) = self.bbox_start

            min_x = min(x,x2)
            max_x = max(x,x2)

            min_y = min(y,y2)
            max_y = max(y,y2)

            x2,y2 = self.bbox_start

            self.rects[self.curr_idx].append([self.annotation_class,min_x,min_y,max_x-min_x,max_y-min_y])

        if self.bbox_active:
            self.select_bbox = [self.bbox_start,(x,y)]

    def __update_image__(self):
        overlay = np.zeros(self.curr_image["segment_img"].shape)

        for (label,x,y,w,h) in self.rects[self.curr_idx]:
            cv2.rectangle(overlay,(x,y),(x+w,y+h),self.label_idx_to_color[label],-1)

        if self.bbox_active:
            (x1,y1),(x2,y2) = self.select_bbox

            min_x = min(x1,x2)
            max_x = max(x1,x2)

            min_y = min(y1,y2)
            max_y = max(y1,y2)
            cv2.rectangle(overlay,(min_x,min_y),(max_x,max_y),(1,0,1,1),-1)

        overlay = cv2.addWeighted(overlay,0.7,overlay,0,0)

        


        self.curr_image["display_img"] = np.clip(cv2.addWeighted(self.curr_image["segment_img"],1,overlay,0.5,0),0,1)

        for (label,x,y,w,h) in self.rects[self.curr_idx]:
            cv2.rectangle(self.curr_image["display_img"],(x,y),(x+w,y+h),self.label_idx_to_color[label],2)

        if self.bbox_active:
            (x1,y1),(x2,y2) = self.select_bbox

            min_x = min(x1,x2)
            max_x = max(x1,x2)

            min_y = min(y1,y2)
            max_y = max(y1,y2)
            cv2.rectangle(self.curr_image["display_img"],(min_x,min_y),(max_x,max_y),(1,0,1,1),2)

        

        if self.curr_idx in self.extra_anno:
            for (x,y) in self.extra_anno[self.curr_idx]:
                cv2.circle(self.curr_image["display_img"],(x,y),5,(1,0,1,1),-1)


        cv2.putText(self.curr_image["display_img"],str(self.curr_idx) + "/" + str(len(self.image_paths)),(0,50),cv2.FONT_HERSHEY_SIMPLEX,2,(1,0,0),2,cv2.LINE_AA)
        cv2.putText(self.curr_image["display_img"],self.label_idx_to_name[self.annotation_class],(0,100),cv2.FONT_HERSHEY_SIMPLEX,2,(1,0,0),2,cv2.LINE_AA)
            


    def __pre_annotate__(self):
        bboxes = self.detector_model.detect(cv2.cvtColor(self.curr_image["org_img"],cv2.COLOR_RGB2BGR),conf = 0.4,model_to_use = self.model_to_use)
        pre_class = self.annotation_class


        for box in bboxes:
            self.annotation_class = int(box.cls.item())
            [x,y,w,h] = box.xywh[0]
            self.__on_click__(1,int((x)/self.curr_image["factor"]),int((y)/self.curr_image["factor"]),None,[False])

        self.annotation_class = pre_class



    
    def __generate_visual_segmentation__(self,anns, file_name,img,org,scalefactor, borders=True):
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

        data = {"org_img": org,"segment_img" : segment_image, "mask" : segment_mask,"factor":scalefactor,"name":file_name}

        np.savez_compressed(file_name,data)
        

            
    
    def process_video(self):
        vid_paths = glob.glob(self.unproc + "*.mp4")

        if len(vid_paths) ==  0:
            print("No videos to process")
            return
        
        img_idx = 0
        for vid_path in tqdm(vid_paths):
            frames_this_vid = 0
            video_capture = cv2.VideoCapture(vid_path)

            frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
            
            saved_frame_name = 0
            bad_shape = False
            for i in tqdm(range(frames)):
                frame_is_read, frame = video_capture.read()

                if frame_is_read:
                    if saved_frame_name % 60 == 0:
                        if frame.shape != (2064,1552,3):
                            if not bad_shape:
                                print("Invalid shape:", frame.shape, "resizing to (2064,1552,3) | Aspect ratio:",max(frame.shape[:2])/min(frame.shape[:2]))


                            frame = cv2.resize(frame,(1552,2064))

                            if not bad_shape:
                                print(frame.shape)
                                bad_shape = True
                        cv2.imwrite(f"{self.unproc}{str(img_idx)}.jpg", frame)
                        img_idx+=1
                        frames_this_vid+=1
                    saved_frame_name += 1
                else:
                    break
            print(f"Saved {frames_this_vid} images from {vid_path}")
            video_capture.release()
            os.rename(vid_path, self.unproc + "raw_finished/" + Path(vid_path).name)
        

        print(f"Saved {img_idx} images in total")



    def process_raw_images(self):
        image_paths = glob.glob(self.unproc + "*.jpg")
        mask_generator = self.__get_mask_generator__()
        curr_highest = 0 if len(glob.glob(self.need_anno+"*.npz")) == 0 else max([int(Path(path).name.split(".")[0]) for path in glob.glob(self.need_anno+"*.npz")]) + 1 
        prev_img_format = (0,0,0)
        for i,img_path in enumerate(tqdm(image_paths)):


            image = load_image(img_path)

            

            scale_factor = max((min(image.shape[:2]) / 512.0),1)


            image_shape = (int(image.shape[1]/scale_factor), int(image.shape[0]/scale_factor))
            scaled_image = cv2.resize(image, image_shape)

            masks = mask_generator.generate(scaled_image)

            file_name = str(i + curr_highest)

            file_name = file_name.zfill(7)

            self.__generate_visual_segmentation__(masks,self.need_anno + img_path.split("\\")[-1][:-4],scaled_image,image,scale_factor)


            prev_img_format = image.shape


    def annotate(self):
        self.image_paths = sorted(glob.glob(self.need_anno + "*.npz"))
        idx = 0
        save = True
        save_override = False


        cv2.namedWindow("image_display")
        cv2.namedWindow("org_img_display")
        cv2.setMouseCallback('image_display', self.__on_click__,[True])

        start_idx = int(read_file("annotate_point.txt")[0])

        self.curr_idx = start_idx

        
        while self.curr_idx < len(self.image_paths):

            self.curr_idx = idx + start_idx

            img_path = self.image_paths[self.curr_idx]

            data = list(np.load(img_path,allow_pickle=True).items())[0][1].item()
            
            image = data["segment_img"]
            org_img = data["org_img"]
            mask = data["mask"]
            other_display  = cv2.resize(data["org_img"],(640,480))

            image[mask == 0] *= 0.3
            image[mask == 0][3] = 1 

            self.curr_image = data

            self.curr_image["display_img"] = image.copy()

            if self.curr_idx not in self.rects:
                self.rects[self.curr_idx] = []
                self.__pre_annotate__()


            while True:
                cv2.imshow("image_display", cv2.cvtColor((data["display_img"]*255).astype(np.uint8),cv2.COLOR_RGB2BGR))
                cv2.imshow("org_img_display", cv2.cvtColor(other_display,cv2.COLOR_RGB2BGR))
                self.__update_image__()
                key_press = cv2.waitKey(1) & 0xff

                if key_press == ord("q"):
                    exit()

                if key_press == ord("r"):
                    self.rects[self.curr_idx].pop()

                if key_press == ord("R"):
                    self.rects[self.curr_idx] = []

                if key_press == ord("a"):
                    idx = max(idx-1,0)
                    break

                if key_press == ord("d"):
                    idx += 1
                    self.annotation_class = 0
                    break

                if key_press == ord("½"):
                    self.annotation_class = 0
                if key_press == ord("1"):
                    self.annotation_class = 1
                if key_press == ord("2"):
                    self.annotation_class = 2
                if key_press == ord("3"):
                    self.annotation_class = 3
                if key_press == ord("4"):
                    self.annotation_class = 4
                if key_press == ord("5"):
                    self.annotation_class = 5
                if key_press == ord("6"):
                    self.annotation_class = 6
                if key_press == ord("7"):
                    self.annotation_class = 7
                if key_press == ord("8"):
                    self.annotation_class = 8
                if key_press == ord("9"):
                    self.annotation_class = 9

                if key_press != 255:
                    print(chr(key_press))


        

            file_name = str(self.curr_idx)

            file_name = file_name.zfill(7)

            labels_string = ""

            for label,x,y,w,h in self.rects[self.curr_idx]:
                labels_string += f"{label} {(x+w/2)/image.shape[1]} {(y+h/2)/image.shape[0]} {w/image.shape[1]} {h/image.shape[0]}\n"

            if len(self.rects[self.curr_idx]) > 0 or save_override:
                if save:
                    labels_string = labels_string.rstrip("\n")
            
                    plt.imsave(f"processed_data/{file_name}.jpg",org_img)

                    print("saving",f"processed_data/{file_name}")

                    with open(f"processed_data/{file_name}.txt","w") as text_file:
                        text_file.write(labels_string)
                    save_override = False
            save = True

            write_file("annotate_point.txt",str(self.curr_idx+1))
    


def create_splits(in_path : str = "processed_data/", out_path : str = "datasets/brick/", splits : list = [("train",0.8),("val",0.2)]):
    random.seed(424354)

    if not Path.exists(Path(in_path)):
        raise IOError(f"in_path doesn't exist: {in_path}")

    if not Path.exists(Path(out_path)):
        Path.mkdir(Path(out_path))
        print(f"out_path doesn't exist: {out_path}, creating it instead")
    

    image_paths = glob.glob(in_path + "*.jpg")
    label_paths = glob.glob(in_path + "*.txt")

    image_paths = sorted(image_paths)
    label_paths = sorted(label_paths)

    splits = sorted(splits,key=lambda x: x[1])


    
    split_count = {name:0 for name,val in splits}

    for i in tqdm(range(len(image_paths))):
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
    anno = Annotator(model_to_use=1)

    anno.process_video()
    # anno.process_raw_images()

    try:
        pass
        anno.annotate()
        # create_splits(in_path="color_processed_data/",out_path="datasets/brick_figures/")
    except Exception as e:
        print(e)

    







