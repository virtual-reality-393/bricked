from brick import *
import glob
import matplotlib.pyplot as plt


annotation_files = glob.glob("processed_data/*.txt")

for file in annotation_files:

    lines = read_file(file)

    result = ""

    for line in lines:

        idx,x,y,w,h = line.replace("\n","").split(" ")

        x = float(x)
        y = float(y)
        h = float(h)
        w = float(w) 

        x = x + w/2
        y = y + h/2
        result += f"{idx} {x} {y} {w} {h}\n"

    write_file(file,result)

        