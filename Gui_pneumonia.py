# Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential, load_model
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import tkinter.font as font
import Gui_graphplot
import Pneumonia_cnn
import demo
import temp_cnn_mri
import DetectBrainTumour

# ccc=1

# Dropdown_frame= tk.Frame(top)
# myfont = font.Font(size=24)


# Dropdown_frame.grid(column=0,row=0,sticky=(N,W,E,S) )
# Dropdown_frame.columnconfigure(0, weight = 1)
# Dropdown_frame.rowconfigure(0, weight = 1)
# Dropdown_frame.pack(pady = 100, padx = 100)
def openfilename(image_name):
    filename = filedialog.askopenfilename(title='open')
    image_name.set(filename)
    return filename


def selectImg(top, image_name):
    img_name = openfilename(image_name)
    img = Image.open(img_name)
    img = img.resize((200, 200), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(top, image=img)
    panel.image = img
    panel.place(relx=0.39, rely=0.3)


def predictImg(top, predicted_output, image_name):
    classifier = temp_cnn_mri.retrieveModel()
    predicted_output.set(temp_cnn_mri.predictSingle(classifier, image_name.get()))
    tk.Label(top, textvariable=predicted_output).grid(row=3, column=0)


def plotGraph(top):
    history = Pneumonia_cnn.loadHistory()
    acc = history['accuracy']
    # print(acc)
    val_acc = history['val_accuracy']

    loss = history['loss']
    val_loss = history['val_loss']
    Gui_graphplot.createWindowforGraph(top, acc, val_acc, loss, val_loss)


def buildClassifier():
    demo.helloCallBack()


# var.set(temp_cnn_mri.helloCallBack())

def generateModel(top):
    Pneumonia_cnn.savingModel()

def getAccuracy(top, accuracy):
    classifier = Pneumonia_cnn.retrieveModel()
    paccuracy = Pneumonia_cnn.getAccuracy(classifier)
    accuracy.set(paccuracy * 100)
    accuracyLabel = tk.Label(top, textvariable=accuracy, height=1, width=15, bg='yellow',font=("Helvetica", 20)).place(relx=0.359, rely=0.122)
    # accuracyLabel['font'] = myfont

def predictPneumoniaImg(top, predicted_output, pneumonia_presence_str, image_name):
    classifier = Pneumonia_cnn.retrieveModel()
    if (Pneumonia_cnn.predictSingle(classifier, image_name.get())):
        # predictImg(predicted_ouput)
        pneumonia_presence_str.set("Pneumonia Present")
        ag = tk.Label(top, textvariable=pneumonia_presence_str, height=2, width=20, bg='red',fg='blue',font=("Helvetica", 20)).place(relx=0.33, rely=0.8)
        # ag['font'] = myfont
    else:
        pneumonia_presence_str.set("Pneumonia not Present")
        predicted_output.set("")
        # tk.Label(top, textvariable=predicted_output).place(relx=0.42, rely=0.16)
        fg = tk.Label(top, textvariable=pneumonia_presence_str, height=2, width=20, bg='red',fg='blue',font=("Helvetica", 20)).place(relx=0.33, rely=0.8)
        # fg['font'] = myfont


def createWindow(master):
    top = tk.Toplevel(master)
    top.title("Pneumonia Detector")
    top.geometry("900x600+450+30")
    image32 = Image.open('saved_model/lungs.png')
    image31 = ImageTk.PhotoImage(image32)
    background_label = tk.Label(top, image=image31)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
    accuracy = tk.IntVar()
    pneumonia_presence_str = tk.StringVar()
    accuracyPneumonia = tk.IntVar()
    image_name = tk.StringVar()
    predicted_output = tk.StringVar()
    create_model = tk.Button(top, text='Generate model', command=lambda: generateModel(top), width=15,
                             height=2).place(relx=0.44, rely=0.04)
    getTypeModuleAccuracyButton = tk.Button(top, text='getAccuracy_of_Type_Module',
                                            command=lambda: getAccuracy(top, accuracy), width=25, height=2)
    # classifier= B.invoke()
    getTypeModuleAccuracyButton.place(relx=0.4, rely=0.23)
    select_Img_button = tk.Button(top, text='select_Img', command=lambda: selectImg(top, image_name), width=15,
                                  height=2)
    select_Img_button.place(relx=0.2, rely=0.4)
    predict_Img_button = tk.Button(top, text='predict_Img',
                                   command=lambda: predictPneumoniaImg(top, predicted_output, pneumonia_presence_str,
                                                                       image_name), width=15, height=2)
    predict_Img_button.place(relx=0.7, rely=0.4)
    plot_graph_button = tk.Button(top, text='plot_graph', command=lambda: plotGraph(top), width=15, height=2)
    plot_graph_button.place(relx=0.44, rely=0.7)
    top.mainloop()
