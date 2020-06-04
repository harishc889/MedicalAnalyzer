import tkinter as tk;
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

def createWindowforGraph(master):
    top=tk.Toplevel(master)
    f = Figure(figsize=(5,5), dpi=100)
    a = f.add_subplot(111)
    a.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])

        

    canvas = FigureCanvasTkAgg(f, top)
    canvas.show()
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    

if __name__=='__main__':
    print("in graph window")