from pyglet.gl import *
from Model import *
import numpy as np
import cv2 as cv
from PIL import Image

class MyWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(400, 300)
        glClearColor(0.2, 0.3, 0.2, 1.0)
        glEnable(GL_DEPTH_TEST)

        self.model = Model('xmas_tree.obj')

    def on_draw(self):
        self.clear()
        glDrawArrays(GL_TRIANGLES, 0, len(self.model.obj.vertex_index))

    def on_resize(self, width, height):
        glViewport(0, 0, width, height)

    def update(self, dt):
        self.model.rotate()



if __name__ == "__main__":
    window = MyWindow(1280, 720, "My Pyglet Window", resizable=True)
    pyglet.clock.schedule_interval(window.update, 1/30.0)
    pyglet.app.run()







