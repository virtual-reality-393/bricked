import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from brick import *
import glob
import math
import matplotlib.pyplot as plt
import random


def onclick(event): 
    image_label.append(1 if event.button == 1 else 0)
    image_points.append((int(event.xdata),int(event.ydata)))
    fig = plt.gcf()
    circle = plt.Circle((event.xdata, event.ydata), radius=10, color='red', fill=False)
    ax = fig.get_axes()[0]

    ax.add_patch(circle)
    fig.canvas.draw()  # Redraw the figure to update the changes

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)
    
DATA_PATH = "unprocessed_data/"
MODEL = "sam2.1_hiera_s"

image_paths = glob.glob(DATA_PATH + "*.jpg")

checkpoint = f"./checkpoints/{MODEL}.pt"
model_cfg = f"configs/sam2.1/{MODEL}.yaml"
sam2 = build_sam2(model_cfg, checkpoint, device="cuda", apply_postprocessing=False)



for i,img_path in enumerate(image_paths):

    
    wm = plt.get_current_fig_manager()
    wm.window.state('zoomed')

    image = load_image(img_path)

    bboxes = []
    image_points = []
    image_label = []


    scale_factor = int(math.floor(min(image.shape[:2]) / 1080.0))
    image_shape = (int(image.shape[1]/scale_factor), int(image.shape[0]/scale_factor))
    image = cv2.resize(image, image_shape)

    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    masks = mask_generator.generate(image)

    ax = plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    fig = ax.get_figure()
    cid = fig.canvas.mpl_connect('button_press_event', onclick) 
    plt.show() 

    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=False)

    for x,y in image_points:
        for mask in sorted_masks:
            if mask["segmentation"][y,x]:
                mask["segmentation"][mask["segmentation"] == True] = False
                bboxes.append(mask["bbox"])
                break

    file_name = str(i)

    file_name = file_name.zfill(7)

    labels_string = ""

    for x,y,w,h in bboxes:
        labels_string += f"0 {x/image.shape[1]} {y/image.shape[0]} {w/image.shape[1]} {h/image.shape[0]}\n"

    labels_string = labels_string.rstrip("\n")
    
    rnd = random.random()
    split = "train" if rnd > 0.2 else "val"


    plt.imsave(f"data/images/{split}/{file_name}.jpg",image)

    with open(f"data/labels/{split}/{file_name}.txt","w") as text_file:
        text_file.write(labels_string)

    sorted_masks = []
    masks = []


    
    






