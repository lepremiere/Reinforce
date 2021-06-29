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
    l1 = l.addLayout(rowspan=1)

    # Reward
    p11 = l1.addPlot(title="Rewards")
    p11.showGrid(x=True, y=True)
    p11.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('w', width=3), movable=False))
    reward_curve = p11.plot(pen=pg.mkPen((252, 152, 3), width=2), name='Reward')
    p11.setXRange(0, schedule[0]*schedule[1] + settings['num_workers'])
    x_reward = []
    y_reward = []
    timer1 = QtCore.QTimer()
    timer1.timeout.connect(lambda: updateCurve(reward_curve,qs[0],x_reward,y_reward))
    timer1.start(10)

    l1.nextRow()
    p12 = l1.addPlot(title="Discounted Rewards")
    p12.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('w', width=3), movable=False))
    p12.showGrid(x=True, y=True)
    discounted_reward_curve = p12.plot(pen=pg.mkPen((252, 152, 3), width=2), name='Discounted Reward')
    p12.setXRange(0, schedule[0]*schedule[1] + settings['num_workers'])
    x_discounted_reward = []
    y_discounted_reward = []
    timer12 = QtCore.QTimer()
    timer12.timeout.connect(lambda: updateCurve(discounted_reward_curve,qs[0],x_discounted_reward,y_discounted_reward))
    timer12.start(10)

    ## Control system
    l2 = l.addLayout(rowspan=1, colspan=2)

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
    p22 = l2.addPlot(title='Train Loss', rowspan=2)
    p22.showGrid(x=True, y=True)
    p22.addLegend(offset=(-10, 10))
    loss_curve_critic = p22.plot(pen=pg.mkPen((0, 3, 252), width=3), name='Critic')
    loss_curve_actor = p22.plot(pen=pg.mkPen((152, 3, 252), width=3), name='Actor')
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
            y_ac.append(item[1][0])
            x_cr.append(item[0])
            y_cr.append(item[1][1])
            curves[0].setData(x_ac,y_ac)
            curves[1].setData(x_cr,y_cr)
        except:
            pass
    timer4 = QtCore.QTimer()
    timer4.timeout.connect(lambda: updateLoss([loss_curve_actor, loss_curve_critic],qs[6],
                                                x_loss_actor,y_loss_actor,x_loss_crtitc,y_loss_critic))
    timer4.start(10)

    # Action Distribution
    l2.nextRow()
    p24 = l2.addPlot(title='Action Distribution')
    p24.showGrid(x=True, y=True)
    
    ## Memory
    l.nextRow()
    l4 = l.addLayout(rowspan=2, colspan=1)

    ## Trading
    l.nextCol()
    l3 = l.addLayout()
    # Return
    p33 = l3.addPlot(title='Return per 24h')
    p33.showGrid(x=True, y=True)
    p33.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('w', width=3), movable=False))
    return_curve = p33.plot(pen=pg.mkPen((235, 64, 52), width=2), name='Reward')
    p33.setXRange(0, schedule[0]*schedule[1] + settings['num_workers'])
    x_return = []
    y_return = []
    timer2 = QtCore.QTimer()
    timer2.timeout.connect(lambda: updateCurve(return_curve,qs[1],x_return,y_return))
    timer2.start(10)

    # Num Trades
    l3.nextRow()
    p31 = l3.addPlot(title='Number of Trades per Hour')
    p31.showGrid(x=True, y=True)
    nt_cuve = p31.plot(pen=pg.mkPen((3, 50, 252), width=2), name='Number of Trades')
    p31.setXRange(0, schedule[0]*schedule[1] + settings['num_workers'])
    x_nt = []
    y_nt = []
    timer5 = QtCore.QTimer()
    timer5.timeout.connect(lambda: updateCurve(nt_cuve,qs[3],x_nt,y_nt))
    timer5.start(50)

    # Average Trade Duration
    l3.nextRow()
    p32 = l3.addPlot(title='Average Trade Duration')
    p32.showGrid(x=True, y=True)
    td_curve = p32.plot(pen=pg.mkPen((3, 219, 252), width=2), name='Trade Duration')
    p32.setXRange(0, schedule[0]*schedule[1] + settings['num_workers'])
    x_td = []
    y_td = []
    timer6 = QtCore.QTimer()
    timer6.timeout.connect(lambda: updateCurve(td_curve,qs[2],x_td,y_td))
    timer6.start(10)

    # Trades
    p34 = l3.addPlot(title="Trades", rowspan=2)
    p34.showGrid(x=True, y=True)
    p34.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('w', width=3), movable=False))
    wins = p34.plot(pen=None, symbolBrush=(11, 156, 3), symbolPen=(11, 156, 3), symbol='t1', symbolSize=12)
    losses = p34.plot(pen=None, symbolBrush=(112, 0, 0), symbolPen=(156, 3, 3), symbol='t', symbolSize=12)
    p34.setXRange(0,500)
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
    timer7.timeout.connect(lambda: updateReturn(p34,wins,losses,qs[5],x_wins,y_wins,x_losses,y_losses))
    timer7.start(5)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

if __name__ == '__main__':
    settings ={'symbol': 'SP500_M5_TA',
                'fraction': [1, 1e5],
                'window_size': 100,
                'num_workers': 30,
                'buffer_size': 40,
                'buffer_batch_size': 1,
                'sort_buffer': True,
                'skewed_memory': True,
                'shuffle_days': True,
                'normalization': False,
                'training_epochs': 5,
                'gamma': 0.99,
                'epsilon': [0.99, 0.999, 0.001],
                'lr_actor': 1e-5,
                'lr_critic': 1e-5,
                'verbose': 1,
                }
    schedule = [1,10,5]
    display('abc', [[0],[1],[2],[3],[4],[5],[6]], settings, schedule)

