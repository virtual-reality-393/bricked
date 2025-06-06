import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter
import cv2
from tqdm import tqdm
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def read_lines_from_file(input_file):
    with open(input_file, "r") as f:
        content = f.read().split("\n")
    return content

def follow_camera(view_ctl, target_center, offset):
    eye = target_center + offset
    up = np.array([0, 1, 0])  # Y-up world
    view_ctl.look_at(target_center, eye, up)

camera_offset = np.array([0, 0, 2]) 
index_to_color = {0:(1,0,0,1),1:(0,1,0,1),2:(0,0,1,1),3:(1,1,0,1),4:(0.4,0.4,1,1),5:(0.4,1,0.4,1),6:(1,1,0.4,1),7:(1,1,1,1),8:(1,0.5,0.5,1),9:(1.0,0.2,0.8,1)}
HAND_BONE_CONNECTIONS = [[0,1],
                         [1,2],[2,3],[3,4],[4,5],[5,19],
                         [1,6],[6,7],[7,8],[8,20],
                         [1,9],[9,10],[10,11],[11,21],
                         [1,12],[12,13],[13,14],[14,22],
                         [1,15],[15,16],[16,17],[17,18],[18,23],
                         [24, 25],
                        [25, 26], [26, 27], [27, 28], [28, 29], [29, 43],
                        [25, 30], [30, 31], [31, 32], [32, 44],
                        [25, 33], [33, 34], [34, 35], [35, 45],
                        [25, 36], [36, 37], [37, 38], [38, 46],
                        [25, 39], [39, 40], [40, 41], [41, 42], [42, 47]
                        ]


lines = read_lines_from_file("game.log")
nameToId = {"red":0.0,"green":1.0,"blue":2.0,"yellow":3.0}

colors = ["red","green","blue","yellow"]

bricks = {}
tableSize = (-1,-1)
points = []

match = re.match(r"\[.*\]",lines[-2])

elements = match.group().split("|")

frameCount = int(elements[2].replace("]",""))

frame_data = [None]*(frameCount+1)

for line in lines:
    match = re.match(r"\[.*\]",line)
    
    if not match:
        print(f"Invalid line: {line}")
        continue

    header_elements = match.group().split("|")

    if len(elements) < 3:
        print(f"Invalid header: {line}")
        continue

    time = int(header_elements[0][1:])

    identifier = header_elements[1]

    frame_num = int(header_elements[2][:-1])

    if frame_data[frame_num] is None:
        frame_data[frame_num] = {}



    input_data = line[match.end():].split(";")


    for data in input_data:
        try:
            type_info,value = data.split(":")

            type, info = type_info.split("_")
        except:
            print(line)


    continue
    
    if "Hand" in identifier:
        if identifier not in frame_data[frame_num]:
            frame_data[frame_num][identifier] = {}
            frame_data[frame_num][identifier]["BONES"] = []
        

        bones = line[match.end():].split(";")

        coords = bones[0].split(":")[1][1:-1].split(",")

        x = float(coords[0])
        y = 1-float(coords[1])

        if 0 < x < 1 and 0 < y < 1:
            frame_data[frame_num][identifier]["PALM"] = (x,y)

        for bone in bones[1:]:
            x,y,z = bone.split(":")[1][1:-1].split(",")
            frame_data[frame_num][identifier]["BONES"].append((float(x),float(y),-float(z)))

    if identifier == "tablePlane":
        elements = line[match.end():].split(";")
        coords = elements[0].split(":")[1].split(",")

        x = float(coords[0])
        y = 1-float(coords[1])
        tableSize = (x,y)

        tablePosition = np.array([float(e) for e in elements[1].split(":")[1][1:-1].split(",")])

        tablePosition[2] *= -1

        tableRotation = np.array([float(e) for e in elements[2].split(":")[1][1:-1].split(",")])


    if identifier == "head":
        elements = line[match.end():].split(";")
        if "FOCAL" in line:
            fx,fy = elements[0].split(":")[1][1:-1].split(",")

            fx = float(fx)
            fy = float(fy)

            cx,cy = elements[1].split(":")[1][1:-1].split(",")

            cx = float(cx)
            cy = float(cy)
        else:
            frame_data[frame_num][identifier] = {}
            x,y,z = elements[0].split(":")[1][1:-1].split(",")
            frame_data[frame_num][identifier]["POSITION"] = (float(x),float(y),-float(z))

            x,y,z,w = elements[1].split(":")[1][1:-1].split(",")
            frame_data[frame_num][identifier]["ROTATION"] = (float(w),float(x),float(y),float(z))
        


    if identifier == "StackGeneration":
        if identifier not in frame_data[frame_num]:
            frame_data[frame_num][identifier] = {}
        
        elements = line[match.end():].split(";")

        event_type = elements[0]

        if event_type == "GENERATE":
            if "POINTS" not in frame_data[frame_num][identifier]:
                frame_data[frame_num][identifier]["POINTS"] = []
            normCoords = elements[2][1:-1].split(",")

            nx = float(normCoords[0])
            ny = 1-float(normCoords[1])

            worldCoords = elements[3][1:-1].split(",")

            wx = float(worldCoords[0])
            wy = float(worldCoords[1])
            wz = -float(worldCoords[2])
            frame_data[frame_num][identifier]["POINTS"].append([(nx,ny,0),(wx,wy,wz),elements[4].split(",")])

           

        if event_type == "COMPLETED":
            if "COMPLETED" not in frame_data[frame_num][identifier]:
                frame_data[frame_num][identifier]["COMPLETED"] = []

            frame_data[frame_num][identifier]["COMPLETED"].append(int(elements[1]))

        if event_type == "FINISHED":
            frame_data[frame_num][identifier]["FINISHED"] = True

        # coords = line[match.end():][1:-1].split(",")

        # x = float(coords[0])
        # y = 1-float(coords[1])

        # frame_data[frameNum][identifier].append([x,y])

    if identifier =="brick":

        values = line[match.end():].split(";")

        coords = values[0][1:-1].split(",")

        x = float(coords[0])
        y = 1-float(coords[1])

        brickType = values[1]

        if brickType not in ["red","green","blue","yellow"]:
            continue

        if frame_num not in bricks:
            bricks[frame_num] = []
        bricks[frame_num].append(np.array([x,y,nameToId[brickType]]))

    # print(frame_data[frameNum])

if tableSize == (-1,-1):
    print("Couldn't get table size, using 1,1")
    tableSize = (1,1)

heatmap_size = (int(250*tableSize[1]), int(250*tableSize[0]))
scale_factor = 4
rescaled_size = np.array([heatmap_size[1],heatmap_size[0]])*scale_factor


out = cv2.VideoWriter('heatmap.mp4', -1, 3, (rescaled_size[0]*2,rescaled_size[1]))

fig = plt.figure()
axs = [fig.add_subplot(211),fig.add_subplot(212)]

right_hand_points = []
right_hand_bone_points = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],]
left_hand_points = []
left_hand_bone_points = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],]
stack_points = []
idx = 0
while frame_data[idx] == None or frame_data[idx] == {}:
    idx+=1
rh_heatmap = np.zeros(heatmap_size)
lh_heatmap = np.zeros(heatmap_size)

vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Point Cloud with Lines',width=int(cx)*2, height=int(cy)*2)

cube = o3d.geometry.TriangleMesh.create_box(width=tableSize[0], height=0.02, depth=tableSize[1])

cube.compute_vertex_normals()

cube.translate(tablePosition - np.array([tableSize[0]/2,0.02,tableSize[1]/2]))

vis.add_geometry(cube)

intrinsics = o3d.camera.PinholeCameraIntrinsic()
intrinsics.set_intrinsics(width=int(cx)*2, height=int(cy)*2, fx=fx/4, fy=fy/4, cx=int(cx), cy=int(cy))

render_option = vis.get_render_option()
render_option.point_size = 15  # Bigger point size
render_option.line_width = 15  # Optional: make lines thicker
# render_option.background_color = np.array([0, 0, 0])  
first = True

line_set = o3d.geometry.LineSet()
pcd = o3d.geometry.PointCloud()

hand_cube_size = 0.01
cubes = []
cube_colors = np.vstack([np.tile(np.array([[1.0, 0.5, 0.0]]), (24, 1)),np.tile(np.array([[0.0, 0.5, 1.0]]), (24, 1)) ])
for i in range(48):
    cubes.append(o3d.geometry.TriangleMesh.create_box(width=hand_cube_size, height=hand_cube_size, depth=hand_cube_size))
    cubes[i].paint_uniform_color((1,0,0) if i < 24 else (0,0,1))
    cubes[i].compute_vertex_normals()

head_geometry = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.2)
vis.add_geometry(head_geometry)

org_head_geometry = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.2)

org_head_geometry.translate(-np.array([0.1,0.1,-0.1]))
pcd.colors = o3d.utility.Vector3dVector()

line_set.lines = o3d.utility.Vector2iVector(HAND_BONE_CONNECTIONS)
line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1.0, 0.0, 1.0]]), (48, 1)) )

extrinsic = np.eye(4)
param = o3d.camera.PinholeCameraParameters()
param.extrinsic = extrinsic
param.intrinsic = intrinsics

view_ctl = vis.get_view_control()
view_ctl.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)


stack_geometry = []

for frame_idx in tqdm(range(idx,frame_num)):
    
    
    curr_frame_data = frame_data[frame_idx]
    org_ctr = vis.get_view_control()
    org_ctr_params = org_ctr.convert_to_pinhole_camera_parameters()

    if "rightHand" in curr_frame_data:

        if "PALM" in curr_frame_data["rightHand"]:
            x,y = curr_frame_data["rightHand"]["PALM"]


            x *= heatmap_size[1]
            y *= heatmap_size[0]

            right_hand_points.append((x,y))

            rh_heatmap[int(y),int(x)] += 1
        if "BONES" in curr_frame_data["rightHand"]: 
            right_hand_bone_points = curr_frame_data["rightHand"]["BONES"]
        

        # line_set = o3d.geometry.LineSet(
        #     points=o3d.utility.Vector3dVector(bonePoints),
        #     lines=o3d.utility.Vector2iVector(HAND_BONE_CONNECTIONS),
        # )
        # # Optional: color lines (here, white)
        # line_colors = np.ones((len(HAND_BONE_CONNECTIONS), 3))
        # line_set.colors = o3d.utility.Vector3dVector(line_colors)

        # Combine and visualize
        

        # Change point size
    
    if "head" in curr_frame_data:
        if "POSITION" in curr_frame_data["head"]:
            rot = R.from_quat(curr_frame_data["head"]["ROTATION"]).as_matrix()
            pos = np.array(curr_frame_data["head"]["POSITION"])

            pos[1] = pos[1]

            R_correction = o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.pi / 2, 0])

            extrinsic[:3,:3] = (rot@R_correction).T
            extrinsic[:3,3] = -rot.T@pos

            # head_geometry.translate(-head_geometry.get_center())

            # head_geometry.translate(pos[1])


            head_geometry.vertices = o3d.utility.Vector3dVector(np.asarray(org_head_geometry.vertices))
            head_geometry.triangles = o3d.utility.Vector3iVector(np.asarray(org_head_geometry.triangles))
            head_geometry.compute_vertex_normals()


            head_geometry.transform(extrinsic)


            vis.update_geometry(head_geometry)

            # param.extrinsic = extrinsic
            # param.intrinsic = intrinsics
            # view_ctl.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
            
      

    if "leftHand" in curr_frame_data:
        if "PALM" in curr_frame_data["leftHand"]:
            x,y = curr_frame_data["leftHand"]["PALM"]

            x *= heatmap_size[1]
            y *= heatmap_size[0]
            left_hand_points.append((x,y))

            lh_heatmap[int(y),int(x)] += 1

        if "BONES" in curr_frame_data["leftHand"]: 
                left_hand_bone_points = curr_frame_data["leftHand"]["BONES"]

    if "StackGeneration" in curr_frame_data:
        if "POINTS" in curr_frame_data["StackGeneration"]:
            stack_points = np.array([e[0] for e in curr_frame_data["StackGeneration"]["POINTS"]])
            stack_points[:,0]*=heatmap_size[1]
            stack_points[:,1]*=heatmap_size[0]


            for i in range(len(curr_frame_data["StackGeneration"]["POINTS"])):
                _,wp,names = curr_frame_data["StackGeneration"]["POINTS"][i]

                for j,name in enumerate(names):

                    cube = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.02, depth=0.02)

                    cube.paint_uniform_color(index_to_color[int(nameToId[name])][:-1])

                    cube.compute_vertex_normals()
                    cube.translate(wp + np.array([0,0.02*(j),0]))

                    vis.add_geometry(cube)

                    stack_geometry.append(cube)




        if "COMPLETED" in curr_frame_data["StackGeneration"]:
            for val in curr_frame_data["StackGeneration"]["COMPLETED"]:
                stack_points[val,2] = 1

        if "FINISHED" in curr_frame_data["StackGeneration"]:
            for geom in stack_geometry:
                vis.remove_geometry(geom)
            
            stack_geometry = []

    rh_np = np.array(right_hand_points)
    lh_np = np.array(left_hand_points)

    
    rh_out = gaussian_filter(rh_heatmap, sigma=4)
    lh_out = gaussian_filter(lh_heatmap, sigma=4)

    rh_out = np.power(rh_out,0.5)
    lh_out = np.power(lh_out,0.5) 

    rh_out = cv2.applyColorMap((rh_out/rh_out.max()*255).astype(np.uint8),cv2.COLORMAP_JET)
    lh_out = cv2.applyColorMap((lh_out/lh_out.max()*255).astype(np.uint8),cv2.COLORMAP_JET)

    rh_out = cv2.resize(rh_out,rescaled_size)
    lh_out = cv2.resize(lh_out,rescaled_size)

    for x,y,c in stack_points:
        x = x*scale_factor
        y = y*scale_factor
        color = (0,255,0) if c == 1 else (0,0,255)
        cv2.circle(rh_out,(int(x),int(y)),25,color,3)
        cv2.circle(rh_out,(int(x),int(y)),22,(0,0,0),1)
        cv2.circle(rh_out,(int(x),int(y)),28,(0,0,0),1)

        cv2.circle(lh_out,(int(x),int(y)),25,color,3)
        cv2.circle(lh_out,(int(x),int(y)),22,(0,0,0),1)
        cv2.circle(lh_out,(int(x),int(y)),28,(0,0,0),1)


    for i in range(1,min(len(right_hand_points),10)):
        x,y = right_hand_points[-i-1]

        x = x*scale_factor
        y = y*scale_factor
        
        cv2.circle(rh_out,(int(x),int(y)),10,(50,0,255),-1)
        cv2.circle(rh_out,(int(x),int(y)),10,(0,0,0),1)

    for i in range(1,min(len(left_hand_points),10)):
        x,y = left_hand_points[-i-1]

        x = x*scale_factor
        y = y*scale_factor
        
        cv2.circle(lh_out,(int(x),int(y)),10,(50,0,255),-1)
        cv2.circle(lh_out,(int(x),int(y)),10,(0,0,0),1)
    
    bonePoints = np.vstack((np.array(right_hand_bone_points),np.array(left_hand_bone_points)))

    line_set.points = o3d.utility.Vector3dVector(bonePoints)
    for i,cube in enumerate(cubes):
        cube.translate(-cube.get_center())
        cube.translate(bonePoints[i])

    if first:
        vis.add_geometry(line_set)
        for cube in cubes:
            vis.add_geometry(cube)
        first = False
    else:
        vis.update_geometry(line_set)
        for cube in cubes:
            vis.update_geometry(cube)
        org_ctr.convert_from_pinhole_camera_parameters(org_ctr_params)
        vis.poll_events()
        vis.update_renderer()


    sep= np.zeros((rescaled_size[1],10,3),dtype=np.uint8)
    final_frame = np.flip(np.hstack((cv2.resize(rh_out,rescaled_size),sep,cv2.resize(lh_out,rescaled_size))),axis=1)
    cv2.imshow("Frame",final_frame)

    if cv2.waitKey(1) == ord("q"):
        break

    # out.write()
    # if len(rh_np.shape) > 1:
    #     for i,x,y in enumerate(rh_np[-10:]):





    #     axs[1].plot(rh_np[-10:,0],rh_np[-10:,1],"m-o",markersize = 1)        
    
    # if len(lh_np.shape) > 1:
    #     axs[0].plot(lh_np[-10:,0],lh_np[-10:,1],"c-o",markersize = 1)
