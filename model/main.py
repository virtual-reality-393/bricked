from brick import *
import glob
import matplotlib.pyplot as plt

DATA_PATH = "datasets/brick_separate/images/train/"

image_paths = sorted(glob.glob("datasets/brick_color_separate/images/train/" + "*.jpg"))
labels = sorted(glob.glob("datasets/brick_color_separate/labels/train/" + "*.txt"))


name_to_index = {"red": 0, "green": 1, "blue": 2, "yellow": 3}


index_to_name = {v: k for k, v in name_to_index.items()}

name_to_color = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0), "magenta": (255, 0, 255)}

def draw_box_color(image, xywh,color):
    x,y,w,h = xywh

    x1 = x
    x2 = x+w
    y1 = y
    y2 = y+h

    cv2.rectangle(
        image, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=color,thickness=5
    )

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
            
            draw_box_color(image,(x1*image.shape[1],y1*image.shape[0],w*image.shape[1],h*image.shape[0]),name_to_color[index_to_name[int(label)]])

    plt.imshow(image)
    plt.show()