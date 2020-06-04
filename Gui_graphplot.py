import matplotlib

matplotlib.use("TkAgg")
import tkinter as tk;
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def createWindowforGraph1(master, fpr, tpr, roc_auc):
    top = tk.Toplevel(master)
    top.title("GraphWindow")
    top.geometry("900x600+450+30")
    f = Figure(figsize=(5, 5), facecolor='white')

    a1 = f.add_subplot(1, 1, 1)
    a1.set_title('ROC Curve')
    t2 = a1.plot(fpr,tpr, color='green', label='AUC=%0.3f' % roc_auc)
    t3 = a1.plot([0,1], [0,1],color='blue')
    a1.set_ylabel('Recall')
    a1.set_xlabel('Fallout(1-specificity)')

    f.legend(loc='lower right')
    canvas = FigureCanvasTkAgg(f, top)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)


def createWindowforGraph(master, accuracy, val_accuracy, loss, val_loss):
    top = tk.Toplevel(master)
    top.title("GraphWindow")
    top.geometry("900x600+450+30")
    f = Figure(figsize=(8, 8), facecolor='white')
    a = f.add_subplot(1, 2, 1)
    a.set_title('Training and Validation Accuracy')
    t0 = a.plot(accuracy, color='red', label='accuracy')
    t1 = a.plot(val_accuracy, color='orange', label='validation accuracy')
    a.set_ylabel('Accuracies')
    a.set_xlabel('Epoch')

    a1 = f.add_subplot(1, 2, 2)
    a1.set_title('Training and Validation Loss')
    t2 = a1.plot(loss, color='green', label='loss')
    t3 = a1.plot(val_loss, color='blue', label='validation_loss')
    a1.set_ylabel('Losses')
    a1.set_xlabel('Epoch')

    f.legend(loc='lower right')
    canvas = FigureCanvasTkAgg(f, top)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)


# =============================================================================
#     toolbar = NavigationToolbar2TkAgg(canvas, top)
#     toolbar.update()
#     canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
# =============================================================================


if __name__ == '__main__':
    print("in graph window")
