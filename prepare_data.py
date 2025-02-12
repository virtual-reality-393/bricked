from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from brick import *
import glob
import math
import matplotlib.pyplot as plt
import cv2
import torch
from pathlib import Path

DATA_PATH = "unprocessed_data/"
MODEL = "sam2.1_hiera_s"

if not Path.exists(Path("unprocessed_data")):
    Path.mkdir(Path("unprocessed_data"))

image_paths = glob.glob(DATA_PATH + "*.jpg")

checkpoint = f"./checkpoints/{MODEL}.pt"
model_cfg = f"configs/sam2.1/{MODEL}.yaml"
sam2 = build_sam2(model_cfg, checkpoint, device="cuda", apply_postprocessing=False)

def show_anns(anns, file_name,img, borders=True):
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
        color_mask = np.concatenate([np.random.random(3), [1]])
        segment_image[m] = segment_image[m]*0.5 + color_mask*0.5
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(segment_image, contours, -1, (0, 0, 1, 0.4), thickness=1)

    data = {"org_img": img,"segment_img" : segment_image, "mask" : segment_mask}
    np.savez_compressed(file_name,data)
    
mask_generator = SAM2AutomaticMaskGenerator(sam2)

for i,img_path in enumerate(image_paths):

    print(f"Processing Image {img_path}")

    image = load_image(img_path)

    bboxes = []
    image_points = []
    image_label = []


    scale_factor = max(int(math.floor(min(image.shape[:2]) / 512.0)),1)
    image_shape = (int(image.shape[1]/scale_factor), int(image.shape[0]/scale_factor))
    image = cv2.resize(image, image_shape)

    
    masks = mask_generator.generate(image)

    file_name = str(i)

    file_name = file_name.zfill(7)

    show_anns(masks,"needs_annotation/" + file_name,image)

    # os.rename(img_path,f"needs_annotation/{file_name}.jpg")

    
    


