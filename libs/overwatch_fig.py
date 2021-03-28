# -*- coding: utf-8 -*-
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pandas as pd
import pyqtgraph as pg
import sys


def display(name, qs):
    app = QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="Нахуи Бааалять!")
    win.resize(1000,600)
    win.setWindowTitle('Overwatch')

    # Enable antialiasing for prettier plots
    pg.setConfigOptions(antialias=True)

    def updateCurve(curve,q,x,y):
        try:
            item = q.get(timeout=0.0)
            q.task_done()
            x.append(item[0])
            y.append(item[1])
            curve.setData(x,y)
        except:
            pass

    # Reward
    p1 = win.addPlot(title="Reward")
    p1.showGrid(x=True, y=True)
    p1.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('w', width=3), movable=False))
    reward_curve = p1.plot(pen=pg.mkPen('y', width=3))
    x_reward = []
    y_reward = []
    timer1 = QtCore.QTimer()
    timer1.timeout.connect(lambda: updateCurve(reward_curve,qs[0],x_reward,y_reward))
    timer1.start(10)

    win.nextRow()

    # Return
    p2 = win.addPlot(title="Return")
    p2.showGrid(x=True, y=True)
    p2.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('w', width=3), movable=False))
    wins = p2.plot(pen=None, symbolBrush='g', symbolPen='g', symbol='t1', symbolSize=12)
    losses = p2.plot(pen=None, symbolBrush='r', symbolPen='r', symbol='t', symbolSize=12)
    p2.setXRange(0,1000)
    x_wins = []
    y_wins = []
    x_losses = []
    y_losses = []
    def updateReturn(p2,wins,losses,q,x_wins,y_wins,x_losses,y_losses):
        try:
            item = q.get(timeout=0.0)
            q.task_done()
            if item[1] > 0:
                if item[0] > 1000:
                    x_wins.pop(0)
                    y_wins.pop(0)
                x_wins.append(item[0])
                y_wins.append(item[1])
                wins.setData(x_wins,y_wins)
            else:
                if item[0] > 1200:
                    x_losses.pop(0)
                    y_losses.pop(0)
                x_losses.append(item[0])
                y_losses.append(item[1])
                losses.setData(x_losses,y_losses)
            if item[0] > 1000:
                p2.setXRange(item[0]-1000,item[0])
        except:
            pass

    timer2 = QtCore.QTimer()
    timer2.timeout.connect(lambda: updateReturn(p2,wins,losses,qs[1],x_wins,y_wins,x_losses,y_losses))
    timer2.start(5)

    win.nextRow()

    # Epsilon
    # p3 = win.addPlot(title="Actions")
    # epsilon_curve = p3.plot(pen='y')
    # x_epsilon = []
    # y_epsilon = []
    # timer3 = QtCore.QTimer()
    # timer3.timeout.connect(lambda: updateCurve(epsilon_curve,qs[2],x_epsilon,y_epsilon))
    # timer3.start(50)

    # p4 = win.addPlot(title="Parametric, grid enabled")
    # x = np.cos(np.linspace(0, 2*np.pi, 1000))
    # y = np.sin(np.linspace(0, 4*np.pi, 1000))
    # # p4.plot(x, y)
    # p4.showGrid(x=True, y=True)

    # p5 = win.addPlot(title="Scatter plot, axis labels, log scale")
    # x = np.random.normal(size=1000) * 1e-5
    # y = x*1000 + 0.005 * np.random.normal(size=1000)
    # y -= y.min()-1.0
    # mask = x > 1e-15
    # x = x[mask]
    # y = y[mask]
    # p5.plot(x, y, pen=None, symbol='t', symbolPen=None, symbolSize=10, symbolBrush=(100, 100, 255, 50))
    # p5.setLabel('left', "Y Axis", units='A')
    # p5.setLabel('bottom', "Y Axis", units='s')
    # p5.setLogMode(x=True, y=False)

    # p6 = win.addPlot(title="Updating plot")
    # curve = p6.plot(pen='y')
    # data = np.random.normal(size=(10,1000))
    # ptr = 0
    # def update():
    #     global curve, data, ptr, p6
    #     curve.setData(data[ptr%10])
    #     if ptr == 0:
    #         p6.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
    #     ptr += 1
    # timer = QtCore.QTimer()
    # timer.timeout.connect(update)
    # timer.start(50)


    # win.nextRow()

    # p7 = win.addPlot(title="Filled plot, axis disabled")
    # y = np.sin(np.linspace(0, 10, 1000)) + np.random.normal(size=1000, scale=0.1)
    # p7.plot(y, fillLevel=-0.3, brush=(50,50,200,100))
    # p7.showAxis('bottom', False)


    # x2 = np.linspace(-100, 100, 1000)
    # data2 = np.sin(x2) / x2
    # p8 = win.addPlot(title="Region Selection")
    # p8.plot(data2, pen=(255,255,255,200))
    # lr = pg.LinearRegionItem([400,700])
    # lr.setZValue(-10)
    # p8.addItem(lr)

    # p9 = win.addPlot(title="Zoom on selected region")
    # p9.plot(data2)
    # def updatePlot():
    #     p9.setXRange(*lr.getRegion(), padding=0)
    # def updateRegion():
    #     lr.setRegion(p9.getViewBox().viewRange()[0])
    # lr.sigRegionChanged.connect(updatePlot)
    # p9.sigXRangeChanged.connect(updateRegion)
    # updatePlot()

## Start Qt event loop unless running in interactive mode or using pyside.

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

if __name__ == '__main__':
    pass
