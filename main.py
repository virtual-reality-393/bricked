import torch
from brick import *
import glob
import matplotlib.pyplot as plt

DATA_PATH = "processed_data/"

image_paths = sorted(glob.glob(DATA_PATH + "*.jpg"))
labels = sorted(glob.glob(DATA_PATH + "*.txt"))



for i in range(len(image_paths)):
    image = load_image(image_paths[i])

    with open(labels[i],"r") as label_file:
        lines = label_file.readlines()

        for line in lines:
            label, x, y, w, h = line.split(" ")

            draw_box(image,(float(x)*image.shape[1],float(y)*image.shape[0],float(w)*image.shape[1],float(h)*image.shape[0]))

    plt.imshow(image)
    plt.show()



