#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""#########################################################
############################################################
### Basic image canvas with pan & zoom in tkinter.       ###
###                                                      ###
### Author:      Daniel Dantas                           ###
### Created:     June   2018                             ###
### Last edited: August 2020                             ###
############################################################
#########################################################"""



## @package ImgCanvas
#  Module with basic image canvas with pan and zoom in tkinter.
#
#  This module implements an image canvas with, pan, zoom and rotation.
#  This module also stores a preview image that can be changed and
#  discarded if necessary.


# python 3
import tkinter as tk

# python 2
# import Tkinter as tk

import my
import math as m
import numpy as np
from PIL import Image
from PIL import ImageTk

WIN_TITLE = "ImgCanvas"

## Class ImgCanvas
#
#  This class implements the image canvas with pan, zoom, rotation and
#  preview capabilities.
class ImgCanvas(tk.Frame):

    ## Object constructor.
    #  @param self The object pointer.
    #  @param root The object root, usualy created by calling tkinter.Tk().
    def __init__(self, root):

        self.none = tk.Frame.__init__(self, root)

        self.root = root
        self.frame = tk.Frame(root)
        self.frame.grid(row=0, column=0, sticky="nsew")

        self.imgScrollVertical = tk.Scrollbar(self.frame, orient="vertical")
        self.imgScrollHorizontal = tk.Scrollbar(self.frame, orient="horizontal")
        self.canvas = tk.Canvas(self.frame, relief=tk.SUNKEN, bd=0, highlightthickness=0,
                                xscrollcommand=self.imgScrollHorizontal.set,
                                yscrollcommand=self.imgScrollVertical.set,
                                scrollregion=(0,0,0,0), cursor="crosshair")
        self.canvas.config(bg="yellow")

        self.imgScrollVertical.config(command=self.canvas.yview)
        self.imgScrollHorizontal.config(command=self.canvas.xview)

        self.imgScrollVertical.pack(side="right", fill="y")
        self.imgScrollHorizontal.pack(side="bottom", fill="x")
        self.canvas.pack(fill='none', expand=True)

        self.canvas.tag_bind("imgTag", "<Motion>", self.cb_on_mouse_move)

        # Zoom

        self.root.bind("<plus>", self.cb_zoom)
        self.root.bind("<minus>", self.cb_zoom)
        self.root.bind("<KP_Add>", self.cb_zoom) #Linux - Numpad
        self.root.bind("<KP_Subtract>", self.cb_zoom) #Linux - Numpad

        #self.root.bind("<Control-Left>", self.cb_rotate_left)
        #self.root.bind("<Control-Right>", self.cb_rotate_right)
        self.root.bind("<k>", self.cb_rotate_left)
        self.root.bind("<l>", self.cb_rotate_right)


        self.image = None
        self.image_save = None

        self.zoom_scale = 1.0
        self.zoom_rate = 2.0
        self.zoom_max = 8.1
        self.zoom_min = 0.124

        self.last_x = 0
        self.last_y = 0

        self.rotate_angle = 0

        # Status

        self.status_label = None
        self.status_format = "X: %d \t Y: %d \t Angle: %3d  Zoom: %.3fx"

        # Pan

        self.canvas.bind("<ButtonPress-3>", self.cb_pan_start)
        self.canvas.bind("<B3-Motion>", self.cb_pan_move)


    """#########################################################
    ############################################################
    ### Canvas functions                                     ###
    ############################################################
    #########################################################"""


    def set_status_label(self, status_label):
        self.status_label = status_label

    def set_status_format(self, status_format):
        self.status_format = status_format

    def set_image(self, image):
        self.image = self.image_rotate(image, self.rotate_angle)
        self.refresh()
        self.status_update()

    def get_image(self):
        return self.image

    def set_preview(self, image):
        if (type(self.image_save) == type(None)):
            self.image_save = self.image
        self.image = image
        self.refresh()
        self.status_update()

    def ok_preview(self):
        self.image_save = None

    def cancel_preview(self):
        if (type(self.image_save) != type(None)):
            self.image = self.image_save
            self.refresh()
            self.status_update()
            self.image_save = None

    def image_size(self, image):
        if (len(image.shape) == 1):
            w = 1
        else:
            w = image.shape[1]
        h = image.shape[0]
        return [w, h]

    def image_rotate(self, image, angle):
        if (angle == 90):
            return np.rot90(image)
        elif (angle == 180):
            aux = np.rot90(image)
            return np.rot90(aux)
        elif (angle == 270):
            aux = np.rot90(image)
            aux = np.rot90(aux)
            return np.rot90(aux)
        return image

    def refresh_no_zoom(self):    
        [w, h] = self.image_size(self.image)
        
        self.canvas.config(width=w, height=h)
        self.canvas.image = ImageTk.PhotoImage(Image.fromarray(self.image))
        self.canvas.imgID = self.canvas.create_image(m.floor(w/2.0), m.floor(h/2.0), image=self.canvas.image, tags="imgTag")
        self.canvas.configure(scrollregion=(0, 0, w, h))

        self.canvas.pack(fill='none', expand=True)


    def refresh(self):
        [w, h] = self.image_size(self.image)

        [w_zoom, h_zoom] = [int(w * self.zoom_scale), int(h * self.zoom_scale)]
        self.size_zoom = [w_zoom, h_zoom]
        self.canvas.config(width=w_zoom, height=h_zoom)

        imgTk = Image.fromarray(self.image)
        self.canvas.image = ImageTk.PhotoImage(imgTk.resize([w_zoom, h_zoom]))
        self.canvas.imgID = self.canvas.create_image(m.floor(w_zoom/2.0), m.floor(h_zoom/2.0), image=self.canvas.image, tags="imgTag")
        self.canvas.configure(scrollregion=(0, 0, w_zoom, h_zoom))

        self.canvas.pack(fill='none', expand=True)

    def zoom_increase(self):
        new_scale = self.zoom_scale * self.zoom_rate
        if( new_scale > self.zoom_max):
            return False
        
        self.zoom_scale = new_scale
        return True

    def zoom_decrease(self):
        new_scale = self.zoom_scale / self.zoom_rate
        if( new_scale < self.zoom_min):
            return False
        
        self.zoom_scale = new_scale
        return True


    """#########################################################
    ############################################################
    ### Callback functions                                   ###
    ############################################################
    #########################################################"""


    def cb_on_mouse_move(self, event):
        if (not self.status_label):
            return

        x = self.canvas.canvasx(event.x) - 1;
        y = self.canvas.canvasy(event.y) - 1;

        self.status_update(x, y)

    def cb_zoom(self, event):
        if( (event.keysym == "plus") or (event.keysym == "KP_Add") ):
            res = self.zoom_increase()
        else:
            res = self.zoom_decrease()

        if(res):
            self.refresh()
            self.status_update()  #updates zoom only

    def cb_pan_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def cb_pan_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def cb_rotate_left(self, event):
        self.image = np.rot90(self.image)
        self.rotate_angle = (self.rotate_angle + 90) % 360
        self.refresh()
        self.status_update()

    def cb_rotate_right(self, event):
        self.image = np.rot90(self.image)
        self.image = np.rot90(self.image)
        self.image = np.rot90(self.image)
        self.rotate_angle = (self.rotate_angle + 270) % 360 
        self.refresh()
        self.status_update()

    def cb_test(self, event=None):
        self.print_event(event)

    def print_event(self, event):
        print(event)
        print("Keysym: " + event.keysym)
        print("Delta:  " + str(event.delta))
        print("Num:    " + str(event.num))
        print("Coord:   (%d, %d)" % (event.x, event.y))


    """#########################################################
    ############################################################
    ### Other functions                                      ###
    ############################################################
    #########################################################"""


    def status_update(self, x=None, y=None):

        if (not x):
            x = self.last_x
        if (not y):
            y = self.last_y
        
        [w_zoom, h_zoom] = self.size_zoom

        #if (x < 0 or x > w_zoom-1 or y < 0 or y > h_zoom-1):
        #    return
        x = self.clamp(x, 0, w_zoom - 1)
        y = self.clamp(y, 0, h_zoom - 1)

        x_zoom = int(x / self.zoom_scale)
        y_zoom = int(y / self.zoom_scale)

        self.last_x = x
        self.last_y = y

        if (self.status_label):
            self.status_label.configure(text=(self.status_format % (x_zoom, y_zoom, self.rotate_angle, self.zoom_scale)))

    def clamp(self, x, x_min, x_max):
        if (x < x_min):
            x = x_min
        if (x > x_max):
            x = x_max
        return x


"""#########################################################
############################################################
### Main function                                        ###
############################################################
#########################################################"""


if __name__ == "__main__":

  root = tk.Tk()
  root.rowconfigure(0, weight=1)
  root.columnconfigure(0, weight=1)
  root.title(WIN_TITLE)
  root.minsize(300,300)

  app = ImgCanvas(root)
  img = my.imread("images/lena.tiff")
  app.set_image(img)

  app.mainloop()

