# -*- coding: utf-8 -*-
from PyQt5.QtCore import qSetFieldWidth
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pandas as pd
import pyqtgraph as pg
import sys


def display(name, qs, settings, schedule):
    app = QtGui.QApplication([])
    l = pg.GraphicsLayout(border=(100,100,100))
    l.setWindowTitle('Overwatch')
    view = pg.GraphicsView()
    view.setCentralItem(l)
    view.show()
    view.resize(1000,600)

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

    # Win Metrics
    l1 = l.addLayout(col=0, row=0, colspan=1, rowspan=1)

    # Reward
    p11 = l1.addPlot(title="Reward and Discounted Rewards", colspan=2)
    p11.showGrid(x=True, y=True)
    p11.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('w', width=3), movable=False))
    reward_curve = p11.plot(pen=pg.mkPen((252, 152, 3), width=3), name='Reward')
    p11.setXRange(0, schedule[0]*schedule[1] + settings['num_workers'])
    x_reward = []
    y_reward = []
    timer1 = QtCore.QTimer()
    timer1.timeout.connect(lambda: updateCurve(reward_curve,qs[0],x_reward,y_reward))
    timer1.start(10)

    # Return
    l1.nextRow()
    p12 = l1.addPlot(title='Return per 24h')
    p12.showGrid(x=True, y=True)
    p12.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('w', width=3), movable=False))
    return_curve = p12.plot(pen=pg.mkPen((235, 64, 52), width=3), name='Reward')
    p12.setXRange(0, schedule[0]*schedule[1] + settings['num_workers'])
    x_return = []
    y_return = []
    timer2 = QtCore.QTimer()
    timer2.timeout.connect(lambda: updateCurve(return_curve,qs[1],x_return,y_return))
    timer2.start(10)

    l.nextCol()

    ## Control system
    l2 = l.addLayout(col=3, row=0, colspan=1, rowspan=1)

    #Epsilon
    p21 = l2.addPlot(title='Epsilon')
    p21.showGrid(x=True, y=True)
    epsilon_cuve = p21.plot(pen=pg.mkPen((3, 128, 252), width=3), name='Epsilon')
    p21.setYRange(0,1)
    p21.setXRange(0, 100)
    x_epsilon = []
    y_epsilon = []
    timer3 = QtCore.QTimer()
    timer3.timeout.connect(lambda: updateCurve(epsilon_cuve,qs[4],x_epsilon,y_epsilon))
    timer3.start(10)
    
    # Loss
    p22 = l2.addPlot(title='Loss', rowspan=2)
    p22.showGrid(x=True, y=True)
    p22.addLegend(offset=(-10, 10))
    loss_curve_actor = p22.plot(pen=pg.mkPen((152, 3, 252), width=3), name='Actor')
    loss_curve_critic = p22.plot(pen=pg.mkPen((0, 3, 252), width=3), name='Critic')
    p22.setXRange(0, schedule[0]*schedule[2] + 1)
    x_loss_actor = []
    y_loss_actor = []
    x_loss_crtitc = []
    y_loss_critic = []
    def updateLoss(curves,q,x_ac,y_ac,x_cr,y_cr):
        try:
            item = q.get(timeout=0.0)
            q.task_done()
            x_ac.append(item[0])
            y_ac.append(item[1])
            x_cr.append(item[0])
            y_cr.appendy(item[2])
            curves[0].setData(x_ac,y_ac)
            curves[1].setData(x_cr,y_cr)
        except:
            pass
    timer4 = QtCore.QTimer()
    timer4.timeout.connect(lambda: updateLoss([loss_curve_actor, loss_curve_critic],qs[6],
                                                x_loss_actor,y_loss_actor,x_loss_crtitc,y_loss_critic))
    timer4.start(10)

    p23 = l2.addPlot(title='Number of Trades per Hour')
    p23.showGrid(x=True, y=True)
    nt_cuve = p23.plot(pen=pg.mkPen((3, 50, 252), width=3), name='Number of Trades')
    p23.setXRange(0, schedule[0]*schedule[1] + settings['num_workers'])
    x_nt = []
    y_nt = []
    timer5 = QtCore.QTimer()
    timer5.timeout.connect(lambda: updateCurve(nt_cuve,qs[3],x_nt,y_nt))
    timer5.start(10)

    # Action Distribution
    l2.nextRow()
    p24 = l2.addPlot(title='Action Distribution')
    p24.showGrid(x=True, y=True)

    # Average Trade Duration
    p25 = l2.addPlot(title='Average Trade Duration')
    p25.showGrid(x=True, y=True)
    td_curve = p25.plot(pen=pg.mkPen((3, 219, 252), width=3), name='Trade Duration')
    p25.setXRange(0, schedule[0]*schedule[1] + settings['num_workers'])
    x_td = []
    y_td = []
    timer6 = QtCore.QTimer()
    timer6.timeout.connect(lambda: updateCurve(td_curve,qs[2],x_td,y_td))
    timer6.start(10)

    # Trades
    l3 = l.addLayout(col=0, row=3, colspan=4, rowspan=1)
    p31 = l3.addPlot(title="Trades")
    p31.showGrid(x=True, y=True)
    p31.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('w', width=3), movable=False))
    wins = p31.plot(pen=None, symbolBrush='g', symbolPen='g', symbol='t1', symbolSize=12)
    losses = p31.plot(pen=None, symbolBrush='r', symbolPen='r', symbol='t', symbolSize=12)
    p31.setXRange(0,500)
    x_wins = []
    y_wins = []
    x_losses = []
    y_losses = []
  
    def updateReturn(p,wins,losses,q,x_wins,y_wins,x_losses,y_losses):
        try:
            n = 500
            # 1/0
            item = q.get(timeout=0.0)
            q.task_done()
            if item[1] > 0:
                if len(x_wins) > n:
                    x_wins.pop(0)
                    y_wins.pop(0)
                x_wins.append(item[0])
                y_wins.append(item[1])
                wins.setData(x_wins,y_wins)
            else:
                if len(x_losses) > n:
                    x_losses.pop(0)
                    y_losses.pop(0)
                x_losses.append(item[0])
                y_losses.append(item[1])
                losses.setData(x_losses,y_losses)
            if item[0] > n:
                p.setXRange(item[0]-n,item[0])
        except:
            pass

    timer7 = QtCore.QTimer()
    timer7.timeout.connect(lambda: updateReturn(p31,wins,losses,qs[5],x_wins,y_wins,x_losses,y_losses))
    timer7.start(5)

    # Epsilon
    # p3 = win.addPlot(title="Actions")
    # epsilon_curve = p3.plot(pen='y')
    # x_epsilon = []
    # y_epsilon = []
    # timer3 = QtCore.QTimer()
    # timer3.timeout.connect(lambda: updateCurve(epsilon_curve,qs[2],x_epsilon,y_epsilon))
    # timer3.start(50)  


    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

if __name__ == '__main__':
    schedule = [1,10,5]
    display('abc', [[1],[2],[3],[4]], schedule)

