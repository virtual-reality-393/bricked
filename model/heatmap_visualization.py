import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter

def read_lines_from_file(input_file):
    with open(input_file, "r") as f:
        content = f.read().split("\n")
    return content


lines = read_lines_from_file("game.log")


nameToId = {"red":0.0,"green":1.0,"blue":2.0,"yellow":3.0}

colors = ["red","green","blue","yellow"]



bricks = {}
tableSize = (-1,-1)
points = []

match = re.match(r"\[.*\]",lines[-2])

elements = match.group().split("|")

frameCount = int(elements[1].replace("]",""))

frame_data = [None]*(frameCount+1)

for line in lines:
    match = re.match(r"\[.*\]",line)
    
    if not match:
        continue

    elements = match.group().split("|")

    if len(elements) < 2:
        continue

    identifier = elements[0].replace("[","").replace("]","")

    frameNum = int(elements[1].replace("]",""))

    if frame_data[frameNum] is None:
        frame_data[frameNum] = {}

    
    if "Hand" in identifier:
        coords = line[match.end():][1:-1].split(",")

        x = float(coords[0])
        y = 1-float(coords[1])

        

        if 0 < x < 1 and 0 < y < 1:
            frame_data[frameNum][identifier] = (x,y)

    if identifier == "tablePlane":
        coords = line[match.end():].split(",")

        x = float(coords[0])
        y = 1-float(coords[1])
        tableSize = (x,y)

    if identifier == "StackGeneration":
        if "COMPLETE" in line:
            continue

        if identifier not in frame_data[frameNum]:
            frame_data[frameNum][identifier] = []

        coords = line[match.end():][1:-1].split(",")

        x = float(coords[0])
        y = 1-float(coords[1])

        frame_data[frameNum][identifier].append([x,y])

    if identifier =="brick":

        values = line[match.end():].split(";")

        coords = values[0][1:-1].split(",")

        x = float(coords[0])
        y = 1-float(coords[1])

        brickType = values[1]

        if brickType not in ["red","green","blue","yellow"]:
            continue

        if frameNum not in bricks:
            bricks[frameNum] = []
        bricks[frameNum].append(np.array([x,y,nameToId[brickType]]))

    # print(frame_data[frameNum])

if tableSize == (-1,-1):
    print("Couldn't get table size, using 1,1")
    tableSize = (1,1)

heatmap_size = (int(200*tableSize[1]), int(200*tableSize[0]))

fig = plt.figure()
axs = [fig.add_subplot(211),fig.add_subplot(212)]

right_hand_points = []
left_hand_points = []
stack_points = []
idx = 0
while frame_data[idx] == None or frame_data[idx] == {}:
    idx+=1

rh_heatmap = np.zeros(heatmap_size)
lh_heatmap = np.zeros(heatmap_size)

def init():
    axs[0].plot([],[])
    axs[1].plot([],[])
    return 

def update(frame_idx):
    global stack_points, right_hand_points, left_hand_points
    frame_idx += idx
    for ax in axs:
        ax.cla()
        ax.set_aspect("equal")
        ax.set_xlim((heatmap_size[1] - 1))
        ax.set_ylim((heatmap_size[0] - 0))
        ax.set_title(frame_idx)

    minVal = max(0,frame_idx-15)
    maxVal = min(len(points),frame_idx)
    
    curr_frame_data = frame_data[frame_idx]

    if curr_frame_data is None:
        return


    if "rightHand" in curr_frame_data:
        x,y = curr_frame_data["rightHand"]

        x *= heatmap_size[1]
        y *= heatmap_size[0]

        right_hand_points.append((x,y))

        rh_heatmap[int(y),int(x)] += 1

    if "leftHand" in curr_frame_data:
        x,y = curr_frame_data["leftHand"]

        x *= heatmap_size[1]
        y *= heatmap_size[0]
        left_hand_points.append((x,y))

        lh_heatmap[int(y),int(x)] += 1

    if "StackGeneration" in curr_frame_data:
        stack_points = np.array(curr_frame_data["StackGeneration"])
    

    rh_np = np.array(right_hand_points)
    lh_np = np.array(left_hand_points)

    
    axs[1].imshow(gaussian_filter(rh_heatmap, sigma=4), cmap='jet', interpolation='nearest')
    axs[0].imshow(gaussian_filter(lh_heatmap, sigma=4), cmap='jet', interpolation='nearest')

    

    axs[0].plot(stack_points[:,0]*heatmap_size[1],stack_points[:,1]*heatmap_size[0],"bo",markersize=10)
    axs[1].plot(stack_points[:,0]*heatmap_size[1],stack_points[:,1]*heatmap_size[0],"bo",markersize=10)

    if len(rh_np.shape) > 1:
        axs[1].plot(rh_np[-10:,0],rh_np[-10:,1],"m-o",markersize = 1)        
    
    if len(lh_np.shape) > 1:
        axs[0].plot(lh_np[-10:,0],lh_np[-10:,1],"c-o",markersize = 1)



ani = animation.FuncAnimation(fig, update, frames=frameCount-idx,
                              init_func=init, blit=False, interval=2)

plt.show()