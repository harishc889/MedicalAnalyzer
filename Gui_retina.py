# Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential, load_model
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

import Gui_graphplot
import Retina_cnn
import demo
import Retina_cnn
import DetectBrainTumour


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


def predictImg(top, image_name, predicted_output):
    classifier = Retina_cnn.retrieveModel()
    predicted_output.set(Retina_cnn.predictSingle(classifier, image_name.get()))
    tk.Label(top, textvariable=predicted_output, height=2, width=20,fg='blue', bg='red').place(relx=0.423, rely=0.8)


def plotGraph(top):
    history = Retina_cnn.loadHistory()
    acc = history['accuracy']
    # print(acc)
    val_acc = history['val_accuracy']

    loss = history['loss']
    val_loss = history['val_loss']
    Gui_graphplot.createWindowforGraph(top, acc, val_acc, loss, val_loss)

def generateModel(top):
    Retina_cnn.savingModel()


def buildClassifier():
    demo.helloCallBack();


# var.set(Retina_cnn.helloCallBack())


def getAccuracy(top, accuracy):
    classifier = Retina_cnn.retrieveModel()
    paccuracy = Retina_cnn.getAccuracy(classifier)
    accuracy.set(paccuracy * 100)
    accuracyLabel = tk.Label(top, textvariable=accuracy, height=2, width=20, fg='blue', bg='orange').place(relx=0.423, rely=0.12)


def predictRetinaImg(top, image_name, predicted_output, retina_presence_str):
    classifier = Retina_cnn.retrieveModel()
    if (Retina_cnn.predictSingle(classifier, image_name.get())):
        predictImg(top, image_name, predicted_output)
        retina_presence_str.set("Retina Present")
        tk.Label(top, textvariable=retina_presence_str, height=2, width=20).place(relx=0.5, rely=0.2)
    else:
        retina_presence_str.set("Retina not Present")


        tk.Label(top, textvariable=retina_presence_str, height=2, width=20).place(relx=0.5, rely=0.2)


def createWindow(master):
    top = tk.Toplevel(master)
    top.title("Retina Defect Detector")
    top.geometry("900x600+450+30")
    image22 = Image.open('saved_model/eye1.png')
    image11 = ImageTk.PhotoImage(image22)
    background_label1 = tk.Label(top, image=image11)
    background_label1.place(x=0, y=0, relwidth=1, relheight=1)

    accuracy = tk.IntVar()
    retina_presence_str = tk.StringVar()
    accuracyRetina = tk.IntVar()
    image_name = tk.StringVar()
    predicted_output = tk.StringVar()
    create_model = tk.Button(top, text='Generate model', command=lambda: generateModel(top), width=15,
                             height=2).place(relx=0.44, rely=0.04)
    getTypeModuleAccuracyButton = tk.Button(top, text='getAccuracy_of_Type_Module',
                                            command=lambda: getAccuracy(top, accuracy), width=25, height=2)
    getTypeModuleAccuracyButton.place(relx=0.4, rely=0.2)
    select_Img_button = tk.Button(top, text='select_Img', command=lambda: selectImg(top, image_name), width=15, height=2)
    select_Img_button.place(relx=0.2, rely=0.4)
    predict_Img_button = tk.Button(top, text='predict_Img',
                                   command=lambda: predictImg(top, image_name, predicted_output), width=15, height=2)
    predict_Img_button.place(relx=0.7, rely=0.4)
    plot_graph_button = tk.Button(top, text='plot_graph', command=lambda: plotGraph(top), width=15, height=2)
    plot_graph_button.place(relx=0.44, rely=0.7)
    top.mainloop()

