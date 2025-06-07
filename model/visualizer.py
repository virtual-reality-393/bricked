import time
from viewer.data_viewer import *
from viewer.geometry import *
from viewer.data_extractor import FrameData
from scipy.spatial.transform import Rotation as R

frame_data = FrameData("game.log")
name_to_color = {
    "red": (1, 0, 0, 1),
    "green": (0, 1, 0, 1),
    "blue": (0, 0, 1, 1),
    "yellow": (1, 1, 0, 1),
}

RIGHT_HAND_BONE_CONNECTIONS = [[0,1],
                         [1,2],[2,3],[3,4],[4,5],[5,19],
                         [1,6],[6,7],[7,8],[8,20],
                         [1,9],[9,10],[10,11],[11,21],
                         [1,12],[12,13],[13,14],[14,22],
                         [1,15],[15,16],[16,17],[17,18],[18,23],
                        ]
LEFT_HAND_BONE_CONNECTIONS = [
                        [24, 25],
                        [25, 26], [26, 27], [27, 28], [28, 29], [29, 43],
                        [25, 30], [30, 31], [31, 32], [32, 44],
                        [25, 33], [33, 34], [34, 35], [35, 45],
                        [25, 36], [36, 37], [37, 38], [38, 46],
                        [25, 39], [39, 40], [40, 41], [41, 42], [42, 47]
]
_,right_hand_data = frame_data.get_data_with_identifier("rightHand",0,True)
_,left_hand_data = frame_data.get_data_with_identifier("leftHand",0,True)
_,table_data = frame_data.get_data_with_identifier("tablePlane",0,True)


data_viewer = DataViewer()
viewer = data_viewer.scene_view

r_boxes = [TriangleGeometry.create_box(f"RBONE{_}", 0.01, 0.01, 0.01) for _ in range(24)]
l_boxes = [TriangleGeometry.create_box(f"LBONE{_}", 0.01, 0.01, 0.01) for _ in range(24)]
table_box = TriangleGeometry.create_box("TABLE",width=table_data["PLANE"][0],height= 0.01, depth = table_data["PLANE"][1])
table_box.geometry.translate(table_data["POSITION"]-np.array([table_data["PLANE"][0]/2,0.005,table_data["PLANE"][1]/2]))

viewer.add_geometry(table_box)
# table_box.set_rotation(R.from_quat(table_data["ROTATION"]).as_matrix())

boxes = r_boxes + l_boxes
r_lines = Lines("RIGHTHANDLINES")
l_lines = Lines("LEFTHANDLINES")
i = 0

for box in r_boxes:
    viewer.add_geometry(box)
    box.set_position(right_hand_data[box.name[1:]])

for box in l_boxes:
    viewer.add_geometry(box)
    box.set_position(left_hand_data[box.name[1:]])

for x,y in RIGHT_HAND_BONE_CONNECTIONS:
    r_lines.add_connection(boxes[x],boxes[y])

for x,y in LEFT_HAND_BONE_CONNECTIONS:
    l_lines.add_connection(boxes[x],boxes[y])

l_lines.update()
r_lines.update()
viewer.add_connections(l_lines)
viewer.add_connections(r_lines)
i=0

stacks = []

newest_event = -1
start = time.time()
curr_time = int((time.time()-start)*1000)
while frame_data.get_closest_frame_to_timestamp(curr_time) != -1:
    curr_time = int((time.time()-start)*1000)
    curr_frame = frame_data.get_closest_frame_to_timestamp(curr_time)
    _, right_hand_data = frame_data.get_data_with_identifier("rightHand", curr_frame)
    _, left_hand_data = frame_data.get_data_with_identifier("leftHand", curr_frame)


    frame, stack_datas = frame_data.get_data_with_identifier("stack", curr_frame)


    if stack_datas is not None and frame > newest_event:
        newest_event = frame
        for stack_data in stack_datas:
            if stack_data["EVENT"] == "GENERATE":
                for i, name in enumerate(stack_data["BRICKS"]):
                    stack_box = TriangleGeometry.create_box(f'Stack{stack_data["STACKNUM"]}_{i}', 0.05, 0.02, 0.02)
                    stack_box.set_color(name_to_color[name])
                    viewer.add_geometry(stack_box)
                    stack_box.set_position(stack_data["POSITION"] + np.array([0, 0.02 * i, 0]))
                    stacks.append(stack_box)

            if stack_data["EVENT"] == "FINISHED":
                for stack in stacks:
                    viewer.remove_geometry(stack.name)

                stacks = []





    if right_hand_data is not None:
        for box in r_boxes:
            box.set_position(right_hand_data[box.name[1:]])

    if left_hand_data is not None:
        for box in l_boxes:
            box.set_position(left_hand_data[box.name[1:]])


    data_viewer.run_one_tick()

data_viewer.quit()