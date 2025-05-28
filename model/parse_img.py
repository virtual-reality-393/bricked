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


contents = read_file("C:\\Users\\VirtualReality\\Documents\\test.txt")



lines = contents.split("\n")

total = []
for line in lines:
    line_numbers = []
    elements = line.split("|")

    for element in elements:
        numbers = element.split(" ")

        numbers = [number.replace(",",".") for number in numbers]

        if len(numbers) == 3:
            line_numbers.append([float(numbers[0]),float(numbers[1]),float(numbers[2])])
        else:
            print(line)
            

    if len(line_numbers) > 0:
        total.append(line_numbers)


total = np.array(total)
print(total.dtype)
plt.imsave("test.png",total)