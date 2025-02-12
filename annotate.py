from brick import *
import glob
import math
import matplotlib.pyplot as plt
import random
import os
from pathlib import Path
mask = []

if not Path.exists(Path("needs_annotation")):
    Path.mkdir(Path("needs_annotation"))
if not Path.exists(Path("processed_data")):
    Path.mkdir(Path("processed_data"))


def onclick(event): 
    image_label.append(1 if event.button == 1 else 0)
    image_points.append((int(event.xdata),int(event.ydata)))
    fig = plt.gcf()
    i = int(event.xdata)
    j = int(event.ydata)

    num,_,flood_mask,rect = cv2.floodFill(mask,np.zeros((mask.shape[0]+2,mask.shape[1]+2),np.uint8),seedPoint=(i,j),newVal=0)

    bboxes.append(rect)

    x,y,w,h = rect

    circle = plt.Rectangle((x,y),w,h, linewidth=4, color='red', fill=False)


    ax = fig.get_axes()[0]
    ax.add_patch(circle)
    fig.canvas.draw()
    
DATA_PATH = "needs_annotation/"

image_paths = glob.glob(DATA_PATH + "*.npy")


for idx,img_path in enumerate(image_paths):

    
    wm = plt.get_current_fig_manager()
    wm.window.state('zoomed')

    data = np.load(img_path,allow_pickle=True).item()


    image = data["segment_img"]
    org_img = data["org_img"]
    mask = data["mask"]

    bboxes = []
    image_points = []
    image_label = []

    ax = plt.imshow(image)
    plt.axis('off')
    fig = ax.get_figure()
    cid = fig.canvas.mpl_connect('button_press_event', onclick) 
    plt.show() 

    file_name = str(idx)

    file_name = file_name.zfill(7)

    labels_string = ""

    for x,y,w,h in bboxes:
        labels_string += f"0 {x/image.shape[1]} {y/image.shape[0]} {w/image.shape[1]} {h/image.shape[0]}\n"

    labels_string = labels_string.rstrip("\n")
    
    rnd = random.random()
    split = "train" if rnd > 0.2 else "val"


    plt.imsave(f"processed_data/{file_name}.jpg",org_img)

    with open(f"processed_data/{file_name}.txt","w") as text_file:
        text_file.write(labels_string)
    
    






