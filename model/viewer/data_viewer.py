from viewer.geometry import *
import open3d as o3d
import open3d.visualization as vis
import numpy as np
import time

class DataViewer:
    def __init__(self,total_time):
        self.app: vis.gui.Application = vis.gui.Application.instance
        self.app.initialize()
        self.window: vis.gui.Window = self.app.create_window("test", 800, 450, 100, 100)
        

        self.horizontal = vis.gui.Horiz()
        self.horizontal.frame = vis.gui.Rect(600, 0, 200, 400)

        self.vert = vis.gui.Vert()
        self.vert.frame = vis.gui.Rect(0, 400, 800, 50)

        self.scene_rect = vis.gui.Rect(0, 0, 600, 400)

        self.scene_view = SceneViewer(self.window.renderer)
        self.scene_view.widget.frame = self.scene_rect
        self.window.add_child(self.scene_view.widget)

        self.settings_view = SettingsViewer()
        self.horizontal.add_child(self.settings_view.widget)

        self.settings_view.enable_table.set_on_checked(
            lambda x: self.scene_view.show_geometry("TABLE") if x else self.scene_view.hide_geometry("TABLE"))

        self.settings_view.enable_left_hand.set_on_checked(lambda x: self.hide_hand(x,"LBONE"))

        self.settings_view.enable_right_hand.set_on_checked(lambda x: self.hide_hand(x,"RBONE"))

        self.settings_view.enable_stacks.set_on_checked(lambda x: self.hide_stacks(x))

        self.video_bar = VideoBar(total_time)
        self.vert.add_child(self.video_bar.widget)
        self.scene_view.widget.set_on_key(lambda x: self.video_bar.toggle(x.type == 0 and not x.is_repeat))

        self.window.add_child(self.horizontal)
        self.window.add_child(self.vert)

    def run_one_tick(self):
        self.scene_view.run_one_tick()
        self.window.post_redraw()
        self.video_bar.update_time()
        return self.app.run_one_tick()

    def get_time(self):
        return self.video_bar.get_time()

    def hide_hand(self,show,identifier):
        [self.scene_view.show_geometry(f"{identifier}{i}") if show else self.scene_view.hide_geometry(f"{identifier}{i}") for i in range(24)]
        side = "RIGHTHANDLINES" if identifier == "RBONE" else "LEFTHANDLINES"
        self.scene_view.show_geometry(side) if show else self.scene_view.hide_geometry(side)

    def hide_stacks(self,show):
        for i in range(8):
            for j in range(8):
                self.scene_view.show_geometry(f"Stack{i}_{j}") if show else self.scene_view.hide_geometry(f"Stack{i}_{j}")

    def set_time_update_callback(self,callback):
        self.video_bar.set_time_callback(callback)


    def quit(self):
        self.app.quit()

    def run(self):
        self.app.run()




class SceneViewer:
    def __init__(self,renderer):
        self.renderer = o3d.visualization.rendering.Open3DScene(renderer)
        self.widget = o3d.visualization.gui.SceneWidget()
        self.widget.scene = self.renderer

        self.widget.scene.set_background([0.2, 0.2, 0.2, 1.0])


        self.connections = []

        self.geometries = {}
        self.hidden_geometries = set()
        self.first_run = True

    def add_geometry(self,geometry : Geometry):
        geometry.scene = self.renderer
        self.geometries[geometry.name] = geometry
        if geometry.name not in self.hidden_geometries:
            self.renderer.add_geometry(geometry.name, geometry.geometry, geometry.material)

    def add_connections(self,lines : Lines):
        self.connections.append(lines)

    def get_bounding_box(self):
        x, y, z = np.array([self.geometries[point].translation for point in self.geometries]).T
        return o3d.geometry.AxisAlignedBoundingBox(np.array([min(x), min(y), min(z)]), np.array([max(x), max(y), max(z)]))

    def hide_geometry(self,name):
        self.hidden_geometries.add(name)
        if name in self.geometries:
            self.renderer.remove_geometry(name)

    def show_geometry(self,name):
        self.hidden_geometries.discard(name)
        if name in self.geometries:
            self.renderer.add_geometry(name,self.geometries[name].geometry,self.geometries[name].material)
            self.geometries[name]._update_geometry()


    def remove_geometry(self, name):
        if name in self.geometries:
            self.geometries.pop(name)
        else:
            print(f"WARNING: Tried to non existing geometry {name}")

        if self.renderer.has_geometry(name):
            self.renderer.remove_geometry(name)

    def run(self):
        bbox = self.get_bounding_box()
        self.widget.setup_camera(60,bbox,[0,0,0])

    def run_one_tick(self):

        if self.first_run:
            bbox = self.get_bounding_box()
            self.widget.setup_camera(60, bbox, [0, 0, 0])
            self.first_run = False

        for connection in self.connections:
            if self.renderer.has_geometry(connection.name):
                self.renderer.remove_geometry(connection.name)
            if connection.name not in self.hidden_geometries:
                connection.update()
                self.renderer.add_geometry(connection.name, connection.line_set, connection.material)






class SettingsViewer:
    def __init__(self):
        self.widget = vis.gui.Vert()

        self.label = vis.gui.Label("Settings")

        self.widget.add_child(self.label)

        self.enable_table = vis.gui.Checkbox("Show Table")
        self.enable_table.checked = True

        self.widget.add_child(self.enable_table)

        self.enable_left_hand = vis.gui.Checkbox("Show Left Hand")
        self.enable_left_hand.checked = True

        self.widget.add_child(self.enable_left_hand)

        self.enable_right_hand = vis.gui.Checkbox("Show Right Hand")
        self.enable_right_hand.checked = True

        self.widget.add_child(self.enable_right_hand)

        self.enable_stacks = vis.gui.Checkbox("Show Stacks")
        self.enable_stacks.checked = True

        self.widget.add_child(self.enable_stacks)



class VideoBar:
    def __init__(self, total_time):
        self.widget = vis.gui.Horiz()
        self.total_time = total_time
        self.slider = vis.gui.Slider(vis.gui.Slider.DOUBLE)
        self.slider.set_limits(0,self.total_time//1000)
        self.widget.add_child(self.slider)
        self.prev_time = time.time()
        self.paused = False
        self.slider.set_on_value_changed(lambda x : self.set_time(x))
        self.time_tracking = 0
        self.callback = None

    def pause(self,state):
        self.paused = state
        self.prev_time = time.time()
        return vis.gui.SceneWidget.EventCallbackResult.HANDLED
    
    def set_time(self, value):
        self.pause(True)
        self.slider.int_value = int(value)
        self.time_tracking = value*1000

        if self.callback is not None:
            self.callback(int(value*1000))

    def update_time(self):
        if not self.paused:
            self.time_tracking += (time.time()-self.prev_time)*1000
            self.slider.int_value = int(self.time_tracking//1000)
            self.prev_time = time.time()

    def set_time_callback(self,callback):
        self.callback = callback

    def toggle(self,toggle):
        if toggle:
            self.paused = not self.paused
            self.prev_time = time.time()
        return vis.gui.SceneWidget.EventCallbackResult.HANDLED

    def get_time(self):
        return int(self.time_tracking)

