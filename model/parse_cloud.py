def read_coords_from_file(input_file):
    with open(input_file, "r") as f:
        content = f.read().replace("\n", "").strip()
    return content

def parse_coords_with_color(coord_string):
    points = []
    for entry in coord_string.split(";"):
        if not entry.strip():
            continue
        try:
            geom_str, color_str = entry.split("|")
            x, y, z, _ = map(float, geom_str.strip("()").split(","))
            r_f, g_f, b_f, _ = map(float, color_str.strip("()").split(","))
            # Convert from float (0.0–1.0) to int (0–255)
            r = int(max(0, min(255, r_f * 255)))
            g = int(max(0, min(255, g_f * 255)))
            b = int(max(0, min(255, b_f * 255)))
            points.append((x, y, z, r, g, b))
        except ValueError as e:
            print(f"Skipping malformed entry: {entry} — {e}")
    return points

def save_as_colored_ply(points, output_file="pointcloud_colored.ply"):
    with open(output_file, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in points:
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
    print(f"Saved {len(points)} colored points to {output_file}")


input_path = "cloud.txt"
coords_string = read_coords_from_file(input_path)
points = parse_coords_with_color(coords_string)
save_as_colored_ply(points, "output_pointcloud_colored.ply")