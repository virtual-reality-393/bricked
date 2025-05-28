import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from collections import defaultdict

# --- CONFIGURATION ---
log_path = "game.log"  # Adjust path to your log file
valid_types = [0, 1, 2, 3]
color_map = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow'}
frame_interval_ms = 50  # Time between frames in milliseconds

# --- LOAD AND PARSE LOG ---
frame_data = defaultdict(list)
pattern = re.compile(r"\[brick-(\d+)\](\d+)\|\((-?\d+\.\d+), (-?\d+\.\d+), (-?\d+\.\d+)\)")

with open(log_path, "r") as file:
    for line in file:
        match = pattern.match(line.strip())
        if match:
            frame = int(match.group(1))
            det_type = int(match.group(2))
            x, y, z = float(match.group(3)), float(match.group(4)), float(match.group(5))
            if det_type in valid_types:
                frame_data[frame].append((det_type, x, y, z))

# --- PREPARE GLOBAL AXIS LIMITS ---
all_x = [x for detections in frame_data.values() for _, x, _, _ in detections]
all_y = [y for detections in frame_data.values() for _, _, y, _ in detections]
all_z = [z for detections in frame_data.values() for _, _, _, z in detections]
x_range, y_range, z_range = (min(all_x), max(all_x)), (min(all_y), max(all_y)), (min(all_z), max(all_z))

sorted_frames = sorted(frame_data.keys())

# --- SETUP PLOT ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatters = {t: ax.plot([], [], [], 'o', color=color_map[t], label=f'Type {t}')[0] for t in valid_types}

def init():
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Detections Over Time (Types 0-3)")
    return scatters.values()

def update(frame_idx):
    frame = sorted_frames[frame_idx]
    detections = frame_data[frame]

    for t in valid_types:
        xs = [x for d_type, x, y, z in detections if d_type == t]
        ys = [y for d_type, x, y, z in detections if d_type == t]
        zs = [z for d_type, x, y, z in detections if d_type == t]
        scatters[t].set_data(xs, ys)
        scatters[t].set_3d_properties(zs)

    ax.set_title(f"3D Detections - Frame {frame}")
    return scatters.values()

# --- ANIMATE ---
ani = animation.FuncAnimation(fig, update, frames=len(sorted_frames),
                              init_func=init, blit=False, interval=frame_interval_ms)

plt.legend()
plt.tight_layout()
plt.show()
