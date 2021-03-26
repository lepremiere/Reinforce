# -*- coding: utf-8 -*-
from pyqtgraph.Qt import QtGui, QtCore
from PyQt5.QtCore import Qt
import numpy as np
import pyqtgraph as pg
from multiprocessing import Process, Queue

# This function is responsible for displaying the data
# it is run in its own process to liberate main process
def display(name,q):
    
    app2 = QtGui.QApplication([])

    win2 = pg.GraphicsWindow(title="Basic plotting examples")
    win2.resize(1000,600)
    win2.setWindowTitle('Nachui bljyat!')
    p2 = win2.addPlot(title="Reward")
    curve = p2.plot(pen='y')

    x_np = []
    y_np = []

    def updateInProc(curve,q,x,y):
        try:
            item = q.get(timeout=0.1)
            x.append(item[0])
            y.append(item[1])
            curve.setData(x,y)
        except:
            pass

    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: updateInProc(curve,q,x_np,y_np))
    timer.start(50)

    QtGui.QApplication.instance().exec_()


if __name__ == '__main__':
    pass