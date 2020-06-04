# This will import all the widgets
# and modules which are available in
# tkinter and ttk module
# from tkinter import *
import tkinter as tk
from tkinter.ttk import *
from tkinter import filedialog, W, N, S, E
from PIL import ImageTk, Image
# import demo
import Gui_Mri_Project
import DetectBrainTumour
import Gui_pneumonia
import Gui_retina
import temp_cnn_mri

# creates a Tk() object

master = tk.Tk()
master.title("Medical Image Analyser")
image2 = Image.open('saved_model/backpic.png')
image1 = ImageTk.PhotoImage(image2)
background_label = tk.Label(master, image=image1)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
accuracy = tk.IntVar()
tumour_presence_str = tk.StringVar()
accuracyTumour = tk.IntVar()
image_name = tk.StringVar()
predicted_output = tk.StringVar()
# sets the geometry of main
# root window
master.geometry("420x400+0+60")



# function to open a new window
# on a button click
def openfilename():
    filename = filedialog.askopenfilename(title='open')
    image_name.set(filename)
    return filename


def openTumourWindow():
    Gui_Mri_Project.createWindow(master)



def openRetinaWindow():
    Gui_retina.createWindow(master)


def openPneumoniaWindow():
    print("fine")

    Gui_pneumonia.createWindow(master)


label = Label(master,
              text="Welcome to Medical Image Analysis Tool.\n\tPlease select a test", anchor="center")

# label.config(width=20)

label.config(font=("Courier", 14, "bold"))
label.grid(row=0, column=0)

# a button widget which will open a
# new window on button click

btn_tumour = Button(master,
                    text="Brain Tumour",
                    command=openTumourWindow)
btn_tumour.grid(row=8, column=0)

btn_retina = Button(master,
                    text="Pneumonia Detection",
                    command=openPneumoniaWindow)
btn_retina.grid(row=14, column=0)

btn_retina = Button(master,
                    text="Retina Defects Detection",
                    command=openRetinaWindow)
btn_retina.grid(row=20, column=0)

# mainloop, runs infinitely

master.mainloop()

if __name__ == '__main__':

    print("hello ji")
