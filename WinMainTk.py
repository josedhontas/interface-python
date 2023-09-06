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


## @package WinMainTk
#  Module with GUIIMP main interface.
#
#  The interface consists of a menu bar, a status bar, a main frame and a
#  toolbox frame.


# python 3
import tkinter as tk
import tkinter.messagebox
import tkinter.filedialog

# python 2
# import Tkinter as tk

import my
import math as m
import os

from PIL import Image
from PIL import ImageTk

import ImgCanvas as ic
import WinThresh
import WinContrast
import WinKernel

WIN_TITLE = "GUIIMP"

## Class WinMainTk
#
#  This class implements the GUIIMP main interface.
class WinMainTk(tk.Frame):

    ## Object constructor.
    #  @param self The object pointer.
    #  @param root The object root, usualy created by calling tkinter.Tk().
    def __init__(self, root):

        self.frame = tk.Frame.__init__(self, root)

        self.root = root

        self.create_bar_menu()

        self.create_frame_main()

        self.create_bar_status()

        self.create_frame_toolbox()


        self.frame_main.set_status_label(self.status)

        self.image_id = 0
        self.image_total = 0
        self.image_filename = []

        self.kernel = my.maskBlur()
        self.strel  = my.seCross3()

        self.bind_all("<comma>", self.cb_shift_z)
        self.bind_all("<period>", self.cb_shift_z)
        self.bind_all("<Left>", self.cb_shift_z)
        self.bind_all("<Right>", self.cb_shift_z)
        self.bind_all("<Next>", self.cb_shift_z)
        self.bind_all("<Prior>", self.cb_shift_z)
        self.bind_all("<MouseWheel>", self.cb_shift_z) # Windows
        self.bind_all("<Button-4>", self.cb_shift_z)   # Linux
        self.bind_all("<Button-5>", self.cb_shift_z)   # Linux

    ## Create menu bar.
    #  @param self The object pointer.
    def create_bar_menu(self):
        self.top = self.winfo_toplevel()        
        self.menu_bar = tk.Menu(self.top)

        self.menu_image = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_image.add_command(label="Image Open", underline=0, command=self.cb_open_image, accelerator="Ctrl+A")
        self.menu_image.add_command(label="Save Image as", underline=5, command=self.dialog_image_save_as, accelerator="Ctrl+S")
        self.menu_image.add_command(label="Open Directory", underline=5, command=self.cb_open_directory, accelerator="Ctrl+P")
        self.menu_image.add_separator()
        self.menu_image.add_command(label="Next", underline=5, command=self.cb_shift_z, accelerator="PgDn")
        self.menu_image.add_command(label="Prior", underline=5, command=self.cb_shift_z, accelerator="PgUp")
        self.menu_image.add_separator()
        self.menu_image.add_command(label="Exit", underline=0, command=self.cb_quit, accelerator="Ctrl+X")
        self.menu_bar.add_cascade(label="Image", underline=0, menu=self.menu_image)

        self.bind_all("<Control-a>", self.cb_open_image)
        self.bind_all("<Control-s>", self.dialog_image_save_as)
        self.bind_all("<Control-p>", self.cb_open_directory)
        self.bind_all("<Control-x>", self.cb_quit)
        
        self.menu_file = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_file.add_command(label="Open", underline=0, command=self.dialog_open, accelerator="Ctrl+O")
        self.menu_file.add_command(label="Save as", underline=5, command=self.dialog_save_as, accelerator="Ctrl+W")
        self.menu_file.add_command(label="Open Directory", underline=5, command=self.dialog_directory, accelerator="Ctrl+D")
        self.menu_file.add_separator()
        self.menu_file.add_command(label="Quit", underline=0, command=self.cb_quit, accelerator="Ctrl+Q")
        self.menu_bar.add_cascade(label="File", underline=0, menu=self.menu_file)

        self.bind_all("<Control-o>", self.dialog_open)
        self.bind_all("<Control-w>", self.dialog_save_as)
        self.bind_all("<Control-d>", self.dialog_directory)
        self.bind_all("<Control-q>", self.cb_quit)
        
        self.menu_message = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_message.add_command(label="Show warning", command=self.box_warning)
        self.menu_message.add_command(label="Show info",  command=self.box_info)
        self.menu_message.add_command(label="Show error", command=self.box_error)
        self.menu_message.add_command(label="Ask question", command=self.box_question)
        self.menu_message.add_command(label="Ask yes/no", command=self.box_yes_no)
        self.menu_message.add_command(label="Ask ok/cancel", command=self.box_ok_cancel)
        self.menu_message.add_command(label="Ask retry/cancel", command=self.box_retry_cancel)
        self.menu_bar.add_cascade(label="Messages", menu=self.menu_message)

        self.menu_help = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_help.add_command(label="About", underline=0, command=self.cb_about, accelerator="F1")
        self.menu_bar.add_cascade(label="Help", menu=self.menu_help)


        self.top.config(menu=self.menu_bar)

    ## Create status bar.
    #  @param self The object pointer.
    def create_bar_status(self):
        self.frame_status = tk.Frame(self.root)
        self.frame_status.grid(row = 1, column = 0, stick='nswe', columnspan=2)

        self.status = tk.Label(self.frame_status, text="", bd=1, relief='sunken', anchor='w', bg='red')
        self.status.pack(side='bottom', fill='x')
        
    ## Create main frame, composed by an ImgCanvas object.
    #  @param self The object pointer.
    def create_frame_main(self):
        self.frame_main = ic.ImgCanvas(self.root)
        self.frame_main.grid(row=0, column=0, stick='nswe')

    ## Create toolbox frame, with buttons to access tools.
    #  @param self The object pointer.
    def create_frame_toolbox(self):
        self.frame_right = tk.Frame(self.root, bg= "orange")
        self.frame_right.grid(row = 0, column = 1, stick='nswe', ipadx=5)

        self.root.columnconfigure(1, weight=0, minsize=200)

        BUTTON_WIDTH = 20

        self.btn_nchannels = tk.Button(self.frame_right, text="nchannels", padx=3, width=BUTTON_WIDTH, command= self.cb_nchannels)
        self.btn_size      = tk.Button(self.frame_right, text="size",      padx=3, width=BUTTON_WIDTH, command= self.cb_size)

        self.btn_rgb2gray  = tk.Button(self.frame_right, text="rgb2gray",  padx=3, width=BUTTON_WIDTH, command= self.cb_rgb2gray)
        self.btn_thresh    = tk.Button(self.frame_right, text="thresh",    padx=3, width=BUTTON_WIDTH, command= self.cb_thresh)

        self.btn_negative  = tk.Button(self.frame_right, text="negative",  padx=3, width=BUTTON_WIDTH, command= self.cb_negative)
        self.btn_contrast  = tk.Button(self.frame_right, text="contrast",  padx=3, width=BUTTON_WIDTH, command= self.cb_contrast)

        self.btn_histeq    = tk.Button(self.frame_right, text="histeq",    padx=3, width=BUTTON_WIDTH, command= self.cb_histeq)
        self.btn_blur      = tk.Button(self.frame_right, text="blur",      padx=3, width=BUTTON_WIDTH, command= self.cb_blur)

        self.btn_kernel    = tk.Button(self.frame_right, text="set convolution kernel",      padx=3, width=BUTTON_WIDTH, command= self.cb_set_kernel)
        
        i = 0
        self.btn_nchannels.grid(row=i, column=0, ipady=5)
        i = i + 1
        self.btn_size.grid(row=i, column=0, ipady=5)

        i = i + 1
        self.btn_rgb2gray.grid(row=i, column=0, ipady=5)
        i = i + 1
        self.btn_thresh.grid(row=i, column=0, ipady=5)

        i = i + 1
        self.btn_negative.grid(row=i, column=0, ipady=5)
        i = i + 1
        self.btn_contrast.grid(row=i, column=0, ipady=5)
        
        i = i + 1
        self.btn_histeq.grid(row=i, column=0, ipady=5)
        i = i + 1
        self.btn_blur.grid(row=i, column=0, ipady=5)

        i = i + 1
        self.btn_kernel.grid(row=i, column=0, ipady=5)


    """#########################################################
    ############################################################
    ### Message box functions                                ###
    ############################################################
    #########################################################"""


    ## Message box to display warning.
    #  @param self The object pointer.
    #  @param msg Warning message to be shown.
    def box_warning(self, msg=None):
        if(not msg):
            msg = "This is a warning.\nPlease click OK button to close."
        tk.messagebox.showwarning("Warning", msg)

    ## Message box to display information.
    #  @param self The object pointer.
    #  @param msg Information message to be shown.
    def box_info(self, msg=None):
        if(not msg):
            msg = "This is an info.\nPlease click OK button to close."
        tk.messagebox.showinfo("Info", msg)

    ## Message box to display error.
    #  @param self The object pointer.
    #  @param msg Error message to be shown.
    def box_error(self, msg=None):
        if(not msg):
            msg = "This is an error.\nPlease click OK button to close."
        tk.messagebox.showerror("Error", msg)

    ## Message box to display yes/no question.
    #  @param self The object pointer.
    def box_question(self):
        ans = tk.messagebox.askquestion("Question", "Choose yes or no", icon = "question")
        self.box_info("You clicked " + ans)

    ## Message box to display yes/no question.
    #  @param self The object pointer.
    def box_yes_no(self):
        ans = tk.messagebox.askyesno("Question", "Choose \"Yes\" or \"No\".", icon = "question")
        msg = "You clicked"  
        if(ans):
            msg = msg + " \"Yes\"."
        else:
            msg = msg + " \"No\"."
        self.box_info(msg)

    ## Message box to display ok/cancel question.
    #  @param self The object pointer.
    def box_ok_cancel(self):
        ans = tk.messagebox.askokcancel("Question", "Choose \"Ok\" or \"Cancel\".", icon = "question")
        msg = "You clicked"  
        if(ans):
            msg = msg + " \"Ok\"."
        else:
            msg = msg + " \"Cancel\"."
        self.box_info(msg)
            
    ## Message box to display retry/cancel question.
    #  @param self The object pointer.
    def box_retry_cancel(self):
        ans = tk.messagebox.askretrycancel("Question", "Choose \"Retry\" or \"Cancel\".", icon = "question")
        msg = "You clicked"  
        if(ans):
            msg = msg + " \"Retry\"."
        else:
            msg = msg + " \"Cancel\"."
        self.box_info(msg)


    """#########################################################
    ############################################################
    ### File dialog functions                                ###
    ############################################################
    #########################################################"""


    SAMPLE_FILETYPES = ( ("Jpeg Images", "*.jpg"),
                        ("Gif Images","*.gif"),
                        ("Png Images","*.png"),
                        ("Tiff Images",".*tiff"),
                        ("All Files", "*.*") )

    SAMPLE_INITIAL_DIR = "/home/ddantas/images"

    def dialog_open(self, event=None, initialdir=SAMPLE_INITIAL_DIR):
        filedir = tk.filedialog.askopenfilename(initialdir=initialdir ,filetypes=self.SAMPLE_FILETYPES)
        if(type(filedir) == str):
            self.box_info("Filename = " + filedir)
        return filedir
                                                            
    def dialog_save_as(self, event=None, initialdir=SAMPLE_INITIAL_DIR):
        filedir = tk.filedialog.asksaveasfilename(initialdir=initialdir ,filetypes=self.SAMPLE_FILETYPES)
        if(type(filedir) == str):
            self.box_info("Filename = " + filedir)
                                                            
    def dialog_directory(self, event=None, initialdir=SAMPLE_INITIAL_DIR):
        filedir = tk.filedialog.askdirectory(initialdir=initialdir)
        if(type(filedir) == str):
            self.box_info("Pathname = " + filedir)
        return filedir


    """#########################################################
    ############################################################
    ### Image dialog functions                               ###
    ############################################################
    #########################################################"""


    FILETYPES_IMAGE = ( ("Jpeg Images", "*.jpg"),
                        ("Gif Images","*.gif"),
                        ("Png Images","*.png"),
                        ("Tiff Images",".*tiff"),
                        ("All Files", "*.*") )

    INITIAL_DIR = "/home/ddantas/images"

    def dialog_image_open(self, event=None, initialdir=INITIAL_DIR):
        filedir = tk.filedialog.askopenfilename(initialdir=initialdir ,filetypes=self.FILETYPES_IMAGE)
        if(type(filedir) == str):
            self.box_info("Filename = " + filedir)
            self.open_image(filedir)
                                                            
    def dialog_image_save_as(self, event=None, initialdir=INITIAL_DIR):
        filedir = tk.filedialog.asksaveasfilename(initialdir=initialdir ,filetypes=self.FILETYPES_IMAGE)
        if(type(filedir) == str):
            self.box_info("Filename = " + filedir)
                                                            
    def dialog_image_directory(self, event=None, initialdir=INITIAL_DIR):
        filedir = tk.filedialog.askdirectory(initialdir=initialdir)
        if(type(filedir) == str):
            self.box_info("Pathname = " + filedir)


    """#########################################################
    ############################################################
    ### Callback functions                                   ###
    ############################################################
    #########################################################"""


    ## Callback: Quit.
    #  Quit program
    #  @param self The object pointer.
    #  @param event The callback event.
    def cb_quit(self, event=None):
        self.root.destroy()

    ## Callback: Change image index.
    #  Callback called when user changes index of image to be shown.
    #  Depending on key pressed, increase or decrease index.
    #  @param self The object pointer.
    #  @param event The callback event.
    def cb_shift_z(self, event=None):
        if   (event.keysym == "comma") or (event.keysym == "Left") or (event.keysym == "Prior") or (event.delta > 0) or (event.num == 4):
            self.cb_shift_z_dec()
        elif (event.keysym == "period") or (event.keysym == "Right") or (event.keysym == "Next") or (event.delta < 0) or (event.num == 5):
            self.cb_shift_z_inc()

    ## Callback: Increase image index.
    #  Callback called when user increases index of image to be shown.
    #  @param self The object pointer.
    #  @param event The callback event.
    def cb_shift_z_inc(self, event=None):
        if self.image_id < self.image_total - 1:
            self.image_id = self.image_id + 1
            self.open_image(self.image_filename[self.image_id])
            self.change_z()

    ## Callback: Decrease image index.
    #  Callback called when user decreases index of image to be shown.
    #  @param self The object pointer.
    #  @param event The callback event.
    def cb_shift_z_dec(self, event=None):
        if( self.image_id  > 0):
            self.image_id = self.image_id - 1
            self.open_image(self.image_filename[self.image_id])
            self.change_z()

    ## Callback: Open single image.
    #  Callback called from menu. Load and display in main frame a single image.
    #  @param self The object pointer.
    #  @param event The callback event.
    def cb_open_image(self, event=None):
        filename = self.dialog_open()
        if (filename):
            self.image_filename = [filename]
            self.open_image(filename)

    ## Callback: Open image directory.
    #  Callback called from menu. Open image directory, load list of file names contained in it,
    #  load and display in main frame first image of directory.
    #  @param self The object pointer.
    #  @param event The callback event.
    def cb_open_directory(self, event=None):
        pathname = self.dialog_directory()
        if (pathname):
            filelist = os.listdir(pathname)
            self.image_filename = filelist
            self.image_total    = len(filelist)
            self.image_id       = 0
            for i in range(len(filelist)):
                self.image_filename[i] = pathname + os.sep + self.image_filename[i]
            print(self.image_filename)
            self.open_image(self.image_filename[0])
            self.change_z()

    ## Callback: show information box about the software.
    #  @param self The object pointer.
    #  @param event The callback event.
    def cb_about(self, event=None):
        msg = "GUIIMP.\nGraphical User Interface for IMage Processing."
        self.box_info(msg)

    ## Change index of image being shown when a whole directory was opened.
    #  Change index of image being shown and update status string.
    #  Change status format string, when a whole directory was opened.
    #  Must be called every time that image index changes.
    #  @param self The object pointer.
    def change_z(self):
        self.frame_main.set_status_format("X: %%d \t Y: %%d \t Z: %d/%d \t Angle: %%3d  Zoom: %%.3fx" % (self.image_id + 1, self.image_total))
        self.frame_main.status_update()

    ## Callback: print event details.
    #  Print event details in console for testing and debugging purpose.
    #  @param self The object pointer.
    #  @param event The event to be printed.
    def cb_test(self, event=None):
        self.print_event(event)

    ## Print event details in console.
    #  Print event details in console for testing and debugging purpose.
    #  @param self The object pointer.
    #  @param event The event to be printed.
    def print_event(self, event):
        print(event)
        print("Keysym: " + event.keysym)
        print("Delta:  " + str(event.delta))
        print("Num:    " + str(event.num))
        print("Coord:   (%d, %d)" % (event.x, event.y))


    """#########################################################
    ############################################################
    ### Toolbox callback functions                           ###
    ############################################################
    #########################################################"""


    ## Callback: show number of image channels.
    #  Callback called when user presses button in toolbox.
    #  Show information box with number of image channels.
    #  @param self The object pointer.
    def cb_nchannels(self):
        msg = "Num. of channels: " + str(my.nchannels(self.get_image()))
        self.box_info(msg)

    ## Callback: show image size.
    #  Callback called when user presses button in toolbox.
    #  Show information box with image size.
    #  @param self The object pointer.
    def cb_size(self):
        msg = "Size: " + str(my.size(self.get_image()))
        self.box_info(msg)

    ## Callback: convert image from RGB to grayscale.
    #  Callback called when user presses button in toolbox.
    #  Convert image from RGB to grayscale and ask for confirmation.
    #  Image must have 3 or more channels.
    #  @param self The object pointer.
    def cb_rgb2gray(self):
        image = self.get_image()
        nchannels = my.nchannels(image)
        if (nchannels < 3):
            return
        result = my.rgb2gray(image)
        self.set_preview(result)
        self.ask_confirmation()

    ## Callback: adjust contrast.
    #  Callback called when user presses button in toolbox.
    #  Open window with threshold parameter. User can change and see modifications in real time
    #  Changes can be committed or cancelled.
    #  @param self The object pointer.
    def cb_thresh(self):
        WinThresh.WinThresh(self.root, self.frame_main)


    ## Callback: negative image.
    #  Callback called when user presses button in toolbox.
    #  Show negative of image in main frame and ask for confirmation.
    #  @param self The object pointer.
    def cb_negative(self):
        image = self.get_image()
        result = my.negative(image)
        self.set_preview(result)
        self.ask_confirmation()

    ## Callback: adjust contrast.
    #  Callback called when user presses button in toolbox.
    #  Open window with contrast parameters. User can change and see modifications in real time
    #  Changes can be committed or cancelled.
    #  @param self The object pointer.
    def cb_contrast(self):
        WinContrast.WinContrast(self.root, self.frame_main)

    ## Callback: histogram equalization.
    #  Callback called when user presses button in toolbox.
    #  Equalize histogram of image in main frame and ask for confirmation.
    #  Image must be grayscale.
    #  @param self The object pointer.
    def cb_histeq(self):
        image = self.get_image()
        nchannels = my.nchannels(image)
        if (nchannels > 1):
            self.box_error("Input must be grayscale")
            return
        result = my.histeq(image)
        self.set_preview(result)
        self.ask_confirmation()

    ## Callback: blur image.
    #  Callback called when user presses button in toolbox.
    #  Blur image in main frame and ask for confirmation.
    #  @param self The object pointer.
    def cb_blur(self):
        image = self.get_image()
        result = my.blur(image)
        self.set_preview(result)
        self.ask_confirmation()

    def cb_set_kernel(self):
        #WinKernel.WinKernel(self.root, self.frame_main, self.kernel)
        WinKernel.WinKernel(self.root, self.frame_main)
        
    ## Ask for confirmation.
    #  Ask for user confirmation whether last change to image in main frame
    #  should be committed or undone.
    #  @param self The object pointer.
    def ask_confirmation(self):
        ans = tk.messagebox.askyesno("Keep changes?", "Keep changes?", icon = "question")
        if(ans):
            self.ok_preview()
        else:
            self.cancel_preview()


    """#########################################################
    ############################################################
    ### Image functions                                      ###
    ############################################################
    #########################################################"""


    ## Open and load single image.
    #  @param self The object pointer.
    #  @param filedir Image full file name.
    def open_image(self, filedir):
        root.title(WIN_TITLE + " - " + filedir)
        try:
            self.image = my.imread(filedir)
        except IsADirectoryError:
            self.box_error("Error: Is a directory:\n%s" % (filedir))
            return
        except:
            self.box_error("Error loading image:\n%s" % (filedir))
            return
        print(self.image)

        try:
            self.frame_main.set_image(self.image)
        except (TypeError):
            self.box_error("Error loading image:\n%s\nPixel format must be uint8." % (filedir))
            return

    ## Get image being shown in main frame.
    #  @param self The object pointer.
    def get_image(self):
        return self.frame_main.get_image()

    ## Set image to be shown in main frame.
    #  @param self The object pointer.
    def set_image(self, image):
        self.frame_main.set_image(image)

    ## Set image to be previewed in main frame.
    #  Preview can be cancelled by calling cancel_preview, or commited by calling ok_preview
    #  @param self The object pointer.
    #  @param filedir Image full file name.
    def set_preview(self, image):
        self.frame_main.set_preview(image)

    ## Commit image being previewed in main frame.
    #  @param self The object pointer.
    def ok_preview(self):
        self.frame_main.ok_preview()

    ## Cancel preview, undoing last modification of image in main frame.
    #  @param self The object pointer.
    def cancel_preview(self):
        self.frame_main.cancel_preview()


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
  
  app = WinMainTk(root)
  app.mainloop()

