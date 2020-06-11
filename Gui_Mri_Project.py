import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

import Gui_graphplot
import demo
import temp_cnn_mri
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


def predictImg(top, predicted_output, image_name):
    classifier = temp_cnn_mri.retrieveModel()
    predicted_output.set(temp_cnn_mri.predictSingle(classifier, image_name.get()))
    tk.Label(top, textvariable=predicted_output, fg='blue', bg='yellow', height=2, width=15).place(relx=0.45, rely=0.9)


def buildClassifier(top):
    demo.helloCallBack();


# var.set(temp_cnn_mri.helloCallBack())

def getAccuracyTumour(top, accuracyTumour):
    model = DetectBrainTumour.load_models()
    taccuracy = DetectBrainTumour.getAccuracy(model)
    accuracyTumour.set(taccuracy * 100)
    tumourAccuracyLabel = tk.Label(top, textvariable=accuracyTumour, fg="blue", bg="orange").place(relx=0.6, rely=0.13)


def getAccuracy(top, accuracy):
    classifier = temp_cnn_mri.retrieveModel()
    paccuracy = temp_cnn_mri.getAccuracy(classifier)
    accuracy.set(paccuracy * 100)
    accuracyLabel = tk.Label(top, textvariable=accuracy, fg="blue", bg="orange").place(relx=0.3, rely=0.13)


def predictTumourImg(top, tumour_presence_str, predicted_output, image_name):
    classifier = DetectBrainTumour.load_models()
    if (DetectBrainTumour.predictTumourPresence(classifier, image_name.get())):
        predictImg(top, predicted_output, image_name)
        tumour_presence_str.set("Tumour Present")
        tk.Label(top, textvariable=tumour_presence_str, fg='blue', bg='yellow', height=2, width=25).place(relx=0.41,
                                                                                                          rely=0.8)
    else:
        tumour_presence_str.set("Tumour not Present")

        tk.Label(top, textvariable=tumour_presence_str, fg='blue', bg='yellow', height=2, width=25).place(relx=0.45, rely=0.8)
        predicted_output.set("")
        tk.Label(top, textvariable=predicted_output, fg='blue', bg='yellow', height=2, width=15).place(relx=0.45,
                                                                                                       rely=0.9)



def plotGraph(top):
    history = temp_cnn_mri.loadHistory()
    acc = history['accuracy']
    # print(acc)
    val_acc = history['val_accuracy']

    loss = history['loss']
    val_loss = history['val_loss']
    Gui_graphplot.createWindowforGraph(top, acc, val_acc, loss, val_loss)


def plot_roc_brain_tumour(top):
    DetectBrainTumour.plot_roc(top);


def generateModel(top):
    temp_cnn_mri.savingModel()


def createWindow(master):
    top = tk.Toplevel(master)
    image222 = Image.open('saved_model/brain2.png')
    image111 = ImageTk.PhotoImage(image222)
    background_label = tk.Label(top, image=image111)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    top.title("Brain Tumour Detector")
    top.geometry("900x600+450+30")

    accuracy = tk.IntVar()
    tumour_presence_str = tk.StringVar()
    accuracyTumour = tk.IntVar()
    image_name = tk.StringVar()
    predicted_output = tk.StringVar()
    create_model = tk.Button(top, text='Generate model', command=lambda: generateModel(top), width=15,
                             height=2).place(relx=0.44, rely=0.04)

    getTypeModuleAccuracyButton = tk.Button(top, text='getAccuracy_of_Type_Module',
                                            command=lambda: getAccuracy(top, accuracy)
                                            , width=25, height=2).place(relx=0.25, rely=0.2)
    # classifier= B.invoke()
    getTumourModuleAccuracyButton = tk.Button(top, text='getAccuracy_of_Tumour_Detect_Module',
                                              command=lambda: getAccuracyTumour(top, accuracyTumour), width=25,
                                              height=2).place(relx=0.55, rely=0.2)
    select_Img_button = tk.Button(top, text='select_Img', command=lambda: selectImg(top, image_name), width=15,
                                  height=2).place(relx=0.2, rely=0.4)
    predict_Img_button = tk.Button(top, text='predict_Img',
                                   command=lambda: predictTumourImg(top, tumour_presence_str, predicted_output,
                                                                    image_name), width=15, height=2).place(relx=0.7,
                                                                                                           rely=0.4)

    plot_graph_button = tk.Button(top, text='plot_graph_mri', command=lambda: plotGraph(top), width=15, height=2).place(
        relx=0.3, rely=0.7)
    plot_roc_button = tk.Button(top, text='plot_roc', command=lambda: plot_roc_brain_tumour(top), width=15,
                                height=2).place(relx=0.59, rely=0.7)

    top.mainloop()


# master1.mainloop()

if __name__ == '__main__':
    print('hello')
