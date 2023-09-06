#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""#########################################################
############################################################
### Basic user interface in tkinter.                     ###
###                                                      ###
### Author:      Daniel Dantas                           ###
### Created:     August 2020                             ###
### Last edited: August 2020                             ###
############################################################
#########################################################"""


## @package WinKernel
#  Module with interface to define a Gaussian kernel.
#
#  The interface consists of a dialog box with text fields to define the
#  kernel horizontal and vertical radii, a text field to define the standard
#  deviation and another to define the amplitude.
#
#  "Update" button recalculates the kernel values.
#
#  "Preview" button applies the convolution with the defined kernel.
#
#  "Ok" button applies the changes to the image.
#
#  "Cancel" button closes the dialog without changing the image.


# python 3
import tkinter as tk

#python 2
#import Tkinter as tk

import my
import math as m
import os

from PIL import Image
from PIL import ImageTk

import ImgCanvas as ic

#mask is ndarray 
import numpy as np

WIN_TITLE = "Kernel"

DEFAULT_RAD = 1
DEFAULT_STDEV = 0.85
DEFAULT_AMP = 0.25
KERNEL_CENTER = 4
MAX_KERNEL_RAD = 4
MAX_KERNEL_DIAM = MAX_KERNEL_RAD * 2 + 1

## Class WinKernel
#
#  This class implements a tk.Frame with an interface to define and edit a
#  Gaussian kernel and to apply it to an image.
class WinKernel(tk.Frame):

    ## Object constructor.
    #  @param self The object pointer.
    #  @param root The object root, usualy created by calling tkinter.Tk().
    def __init__(self, root, canvas=None):

        self.root = root
        self.top = tk.Toplevel(self.root)
        self.top.title(WIN_TITLE + " - parameters")

        self.none = tk.Frame.__init__(self, self.top)
        
        self.canvas = canvas
        self.image = self.canvas.get_image()

        # Frame up: buttons
        self.frame_up = tk.Frame(self.top)
        self.frame_up.grid(row = 0, column = 0, stick='nswe', ipadx=5)

        self.l1 = tk.Label(self.frame_up, text="Horizontal radius", padx=0)
        self.l1.grid(row=0, column=0, ipady=5)

        self.e1 = tk.Entry(self.frame_up)
        self.e1.grid(row=0, column=1, ipady=5)
        self.e1.insert(0, str(DEFAULT_RAD))

        self.l2 = tk.Label(self.frame_up, text="Vertical radius", padx=00)
        self.l2.grid(row=1, column=0, ipady=5)

        self.e2 = tk.Entry(self.frame_up)
        self.e2.grid(row=1, column=1, ipady=5)
        self.e2.insert(0, str(DEFAULT_RAD))

        self.b0 = tk.Button(self.frame_up, text="Change size", command=self.cb_resize)
        self.b0.grid(row=2, column=0, ipady=5)

        self.l3 = tk.Label(self.frame_up, text="Standard deviation", padx=00)
        self.l3.grid(row=3, column=0, ipady=5)

        self.e3 = tk.Entry(self.frame_up)
        self.e3.grid(row=3, column=1, ipady=5)
        self.e3.insert(0, str(DEFAULT_STDEV))

        self.l4 = tk.Label(self.frame_up, text="Amplitude", padx=00)
        self.l4.grid(row=4, column=0, ipady=5)

        self.e4 = tk.Entry(self.frame_up)
        self.e4.grid(row=4, column=1, ipady=5)
        self.e4.insert(0, str(DEFAULT_AMP))

        self.b3 = tk.Button(self.frame_up, text="Update", command=self.cb_update)
        self.b3.grid(row=5, column=0, ipady=5)

        self.b4 = tk.Button(self.frame_up, text="Preview", command=self.cb_preview)
        self.b4.grid(row=5, column=1, ipady=5)

        self.b1 = tk.Button(self.frame_up, text="Ok", command=self.cb_ok)
        self.b1.grid(row=6, column=0, ipady=5)

        self.b2 = tk.Button(self.frame_up, text="Cancel", command=self.cb_cancel)
        self.b2.grid(row=6, column=1, ipady=5)

        # Frame down: values
        self.frame_down = tk.Frame(self.top)
        self.frame_down.grid(row = 1, column = 0, stick='nswe', ipadx=5)

        self.kernel_entry = []
        for i in range(MAX_KERNEL_DIAM):
            for j in range(MAX_KERNEL_DIAM):
                idx = i * MAX_KERNEL_DIAM + j
                self.kernel_entry.append(tk.Entry(self.frame_down, width=4))
                self.kernel_entry[idx].grid(row=i, column=j, ipady=5)

        self.cb_update(self)

    def gaussian(self, i, j, stdev):
        di = i - KERNEL_CENTER
        dj = j - KERNEL_CENTER
        return m.exp(   - ( (di*di) / (2.0*stdev*stdev) + (dj*dj) / (2.0*stdev*stdev) )   )

    def get_mask(self):
        rh = int(self.e1.get())
        rv = int(self.e2.get())
        w = 2 * rh + 1
        h = 2 * rv + 1
        mask = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                i2 = KERNEL_CENTER - rv + i
                j2 = KERNEL_CENTER - rh + j
                idx = i2 * MAX_KERNEL_DIAM + j2
                mask[i, j] = self.kernel_entry[idx].get()
        print(mask)
        return mask
                

    """#########################################################
    ############################################################
    ### Callback functions                                   ###
    ############################################################
    #########################################################"""


    def cb_resize(self, event=None):
        rh = int(self.e1.get())
        rv = int(self.e2.get())
        for i in range(MAX_KERNEL_DIAM):
            for j in range(MAX_KERNEL_DIAM):
                idx = i * MAX_KERNEL_DIAM + j
                if ( (i < KERNEL_CENTER - rv) or (i > KERNEL_CENTER + rv) or
                     (j < KERNEL_CENTER - rh) or (j > KERNEL_CENTER + rh) ):
                    self.kernel_entry[idx].config(state=tk.DISABLED)
                else:
                    self.kernel_entry[idx].config(state=tk.NORMAL)

    def cb_update(self, event=None):
        stdev = float(self.e3.get())
        amp   = float(self.e4.get())
        for i in range(MAX_KERNEL_DIAM):
            for j in range(MAX_KERNEL_DIAM):
                idx = i * MAX_KERNEL_DIAM + j
                self.kernel_entry[idx].config(state=tk.NORMAL)
                w = self.gaussian(i, j, stdev) * amp
                self.kernel_entry[idx].delete(0, tk.END)
                self.kernel_entry[idx].insert(0, str(round(w, 2)))
        self.cb_resize(self)

    def cb_preview(self, event=None):
        mask = self.get_mask()
        result = my.convolve(self.image, mask)
        self.canvas.set_preview(result)
        print("Done.")

    def cb_ok(self, event=None):
        self.canvas.ok_preview()
        self.top.destroy()

    def cb_cancel(self, event=None):
        self.canvas.cancel_preview()
        self.top.destroy()

        


############################################################
# Main function
############################################################

if __name__ == "__main__":

  root = tk.Tk()
  root.rowconfigure(0, weight=1)
  root.columnconfigure(0, weight=1)
  root.title(WIN_TITLE)
  root.minsize(600, 270)

  """  
  win = tk.Toplevel(root)

  root.frame_main = ic.ImgCanvas(root)
  root.frame_main.grid(row=0, column=0, stick='nswe', ipadx=5)

  root.frame_toolbox = tk.Frame(root, bg='orange')
  root.frame_toolbox.grid(row=0, column=1, stick='nswe', ipadx=5)

  #win.title = "ImgCanvas"
  img = my.imread("images/lena.tiff")
  root.frame_main.set_image(img)
  """

  img = my.imread("images/lena256.tiff")
  root.ic = ic.ImgCanvas(root)
  root.ic.set_image(img)

  app = WinKernel(root, root.ic)
  app.mainloop()

