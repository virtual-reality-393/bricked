import pygetwindow as gw
from PIL import Image
from io import BytesIO
import glob
from pathlib import Path
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_file(fn):
    contents = ""
    with open(fn,"r") as text_file:
        contents = text_file.read()
    return contents


contents = read_file("C:\\Users\\VirtualReality\\Documents\\log.txt")



lines = contents.split("\n")


line_idx = [i for i in range(len(lines)) if "line" in lines[i]]

start_idx = [i for i in line_idx if "line 0" in lines[i]][0]

coords = []
coord_set = []
i = start_idx
while i < len(lines):
    if "line" in lines[i]:
        if(len(coord_set) > 0):
            coords.append(coord_set)
        coord_set = []

    if "|" in lines[i]:
        pixels = lines[i].split(":")[-1]

        elements = pixels.split("|")

        for element in elements:
            element = element.strip()
            if len(element) == 0:
                continue

            if len(element.split(" ")) != 3:
                print(element)
            coord_set.append([float(num) for num in element.split(" ")])
    i+=1

coords.append(coord_set)


print([i for i in range(len(coords)) if len(coords[i]) != 640])
coords.pop(430)
img = np.array(coords)

# img = img ** (1/2.2)

plt.imsave("test.png",img)