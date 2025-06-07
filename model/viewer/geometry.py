import open3d as o3d
import open3d.visualization as vis
import numpy as np
from copy import deepcopy

class Geometry:
    def __init__(self,geometry : o3d.geometry.Geometry3D, name, material : vis.rendering.MaterialRecord = None):
        self.geometry = geometry
        self.translation = np.zeros(3)
        self.rotation = np.eye(3)
        self.name = name
        self.scene : o3d.visualization.rendering.Open3DScene = ""

        if material is None:
            self.material = vis.rendering.MaterialRecord()
            self.material.shader = "defaultLit"
        else:
            self.material = material

    def set_position(self,position):
        self.translation = np.array(position)
        self._update_geometry()

    def set_rotation(self,rotation):
        self.rotation = np.array(rotation)
        self._update_geometry()

    def set_color(self,color):
        self.material.base_color = color

    def _update_geometry(self):
        if self.scene == "":
            print(f"WARNING: Attempting to update geometry {self.name} with no scene")
            return

        transform_matrix = np.eye(4)
        transform_matrix[:3,:3] = self.rotation.T
        transform_matrix[:3,3] = self.translation@self.rotation.T
        self.scene.set_geometry_transform(self.name,transform_matrix)

    def __hash__(self):
        return hash(self.name)

class TriangleGeometry(Geometry):
    def __init__(self,geometry : o3d.geometry.TriangleMesh, name):
        super().__init__(geometry,name)
        self.geometry.compute_vertex_normals()
        self.org_mesh = deepcopy(geometry)


    def set_rotation(self,rotation):
        self.geometry = deepcopy(self.org_mesh)
        super().set_rotation(rotation)

    @classmethod
    def create_box(cls,name,width,height,depth):
        return TriangleGeometry(o3d.geometry.TriangleMesh.create_box(width,height,depth),name)

class Lines:
    def __init__(self,name):
        self.connections = []
        self.points = {}
        self.name = name
        self.line_set = o3d.geometry.LineSet()
        self.material = vis.rendering.MaterialRecord()
        self.material.shader = "unlitLine"
        self.material.line_width = 10
        # self.material.base_color = np.array([1,1,1,1])

    def add_connection(self,geom1 : Geometry,geom2 : Geometry):
        if geom1 not in self.points:
            self.points[geom1] = len(self.points)

        if geom2 not in self.points:
            self.points[geom2] = len(self.points)

        self.connections.append((geom1,geom2))

    def construct(self):
        self.line_set = o3d.geometry.LineSet()

        cons = []
        for geom1, geom2 in self.connections:
            cons.append([self.points[geom1], self.points[geom2]])
        self.line_set.points = o3d.utility.Vector3dVector(
            [point.translation + point.geometry.get_center() for point in self.points])

        self.line_set.lines = o3d.utility.Vector2iVector(cons)

    def update(self):
        self.construct()



