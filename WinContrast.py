#!/usr/bin/python3
# -*- coding: utf-8 -*-


"""#########################################################
############################################################
### Basic user interface in tkinter.                     ###
###                                                      ###
### Author:      Daniel Dantas                           ###
### Created:     June   2018                             ###
### Last edited: August 2020                             ###
############################################################
#########################################################"""


## @package WinContrast
#  Module with interface to change image contrast.
#
#  The interface consists of a dialog box with two scales: one for the angular
#  coefficient and another for the linear coefficient. There are also an "Ok"
#  button that applies the changes and a "Cancel" button that closes the dialog.


# python 3
import tkinter as tk

# python 2
# import Tkinter as tk

import my
import math as m
import os

from PIL import Image
from PIL import ImageTk

import ImgCanvas as ic

WIN_TITLE = "Contrast"

## Class WinContrast
#
#  This class implements a tk.Frame with an interface to change the
#  contrast of an image stored in a canvas.
class WinContrast(tk.Frame):

    ## Object constructor.
    #  @param self The object pointer.
    #  @param root The object root, usualy created by calling tkinter.Tk().
    #  @param canvas The object canvas that stores the image.
    def __init__(self, root, canvas=None):

        self.root = root
        self.top = tk.Toplevel(self.root)
        self.top.title(WIN_TITLE + " - parameters")
        
        self.none = tk.Frame.__init__(self, self.top)
        
        self.canvas = canvas
        self.image = self.canvas.get_image()

        self.l1 = tk.Label(self.top, text="Angular coefficient r", pady=20)
        self.l1.pack()

        self.s1 = tk.Scale(self.top, from_=0, to=5, length=512, tickinterval=0.5, orient=tk.HORIZONTAL, command=self.cb_contrast, resolution=0.01)
        self.s1.set(1.0)
        self.s1.pack()

        self.l2 = tk.Label(self.top, text="Linear coefficient m", pady=20)
        self.l2.pack()

        self.s2 = tk.Scale(self.top, from_=0, to=255, length=512, tickinterval=16, orient=tk.HORIZONTAL, command=self.cb_contrast)
        self.s2.set(128)
        self.s2.pack()

        self.f1 = tk.Frame(self.top)

        self.b1 = tk.Button(self.f1, text="Ok", command=self.cb_ok)
        self.b1.pack(side=tk.LEFT) 

        self.b2 = tk.Button(self.f1, text="Cancel", command=self.cb_cancel)
        self.b2.pack(side=tk.LEFT)

        self.f1.pack()


    """#########################################################
    ############################################################
    ### Callback functions                                   ###
    ############################################################
    #########################################################"""


    ## Callback: Change contrast parameters and update preview.
    #  Callback called when user changes slider value. Contrast parameters are changed and result is previewed.
    #  @param self The object pointer.
    #  @param event The callback event.
    def cb_contrast(self, event=None):
        print("%f, %d" % (self.s1.get(), self.s2.get()))
        r = self.s1.get()
        m = self.s2.get()
        result = my.contrast(self.image, r, m)
        self.canvas.set_preview(result)

    ## Callback: Commit previewed image.
    #  Callback called when user presses Ok button. Previewed result is commited.
    #  @param self The object pointer.
    #  @param event The callback event.
    def cb_ok(self, event=None):
        self.canvas.ok_preview()
        self.top.destroy()

    ## Callback: Cancel previewed image.
    #  Callback called when user presses Cancel button. Previewed result is discarded.
    #  @param self The object pointer.
    #  @param event The callback event.
    def cb_cancel(self, event=None):
        self.canvas.cancel_preview()
        self.top.destroy()

        
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
  root.minsize(600, 270)

  img = my.imread("images/lena.tiff")
  root.ic = ic.ImgCanvas(root)
  root.ic.set_image(img)

  app = WinContrast(root, root.ic)
  app.mainloop()

