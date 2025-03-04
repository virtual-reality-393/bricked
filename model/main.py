from brick import *
import glob
import matplotlib.pyplot as plt

DATA_PATH = "processed_data/"

image_paths = sorted(glob.glob(DATA_PATH + "*.jpg"))[-1:]
labels = sorted(glob.glob(DATA_PATH + "*.txt"))[-1:]

for i in range(len(image_paths)):
    image = load_image(image_paths[i])

    with open(labels[i],"r") as label_file:
        lines = label_file.readlines()

        for line in lines:
            label, x, y, w, h = line.split(" ")

            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)

            x1 = x-w/2
            y1 = y-h/2
            
            draw_box(image,(x1*image.shape[1],y1*image.shape[0],w*image.shape[1],h*image.shape[0]))

    plt.imshow(image)
    plt.show()