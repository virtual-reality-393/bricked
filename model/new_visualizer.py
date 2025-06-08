import open3d as o3d
print(o3d.__version__)
# Setup Application (required before using O3DVisualizer)
app = o3d.visualization.gui.Application.instance
app.initialize()

# Create a PointCloud and LineSet
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0]])
pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0]])

lines = [[0, 1]]
line_set = o3d.geometry.LineSet(
    points=pcd.points,
    lines=o3d.utility.Vector2iVector(lines),
)

# Create custom material for the LineSet
line_material = o3d.visualization.rendering.MaterialRecord()
line_material.shader = "unlitLine"
line_material.line_width = 5.0  # Set line width here

# Create material for the PointCloud (optional)
pcd_material = o3d.visualization.rendering.MaterialRecord()
pcd_material.shader = "defaultUnlit"

# Create a window and scene widget
window = app.create_window("Line Width Example", 1024, 768)
scene = o3d.visualization.gui.SceneWidget()
scene.scene = o3d.visualization.rendering.Open3DScene(window.renderer)

# Add geometries
scene.scene.add_geometry("PointCloud", pcd, pcd_material)
scene.scene.add_geometry("LineSet", line_set, line_material)

# Add scene to the window
window.add_child(scene)

# Run the app
app.run()


app.close()
