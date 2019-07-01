#! encoding = utf-8

''' Demostrate lissajour curve '''

import sys
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg



class MainWindow(QtWidgets.QMainWindow):
    '''
        Implements the main window
    '''

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self)
        self.setStyleSheet('font-size: 12pt; font-family: default')
        self.setWindowTitle('Lissajour Curve Demo')

        self.canvasBox = CanvasBox(self)
        self.parBox = ParBox(self)

        mainLayout = QtWidgets.QHBoxLayout()
        mainLayout.addWidget(self.canvasBox)
        mainLayout.addWidget(self.parBox)
        # Enable main window
        mainWidget = QtWidgets.QWidget()
        mainWidget.setLayout(mainLayout)
        self.setCentralWidget(mainWidget)



class ParBox(QtWidgets.QGroupBox):

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self)
        self.parent = parent
        self.t = np.linspace(0, 2*np.pi, 1000)
        self.x = np.zeros_like(self.t)
        self.y = np.zeros_like(self.t)

        label1 = QtWidgets.QLabel('x = cos(p*t)')
        label1.setStyleSheet('font-size: 14pt')
        label2 = QtWidgets.QLabel('y = cos(q*t + φ)')
        label2.setStyleSheet('font-size: 14pt')

        self.linkpqCheck = QtWidgets.QCheckBox('Link p, q')

        self.pInput = QtWidgets.QSpinBox()
        self.pInput.setPrefix('p: ')
        self.pInput.setValue(1)

        self.qInput = QtWidgets.QSpinBox()
        self.qInput.setPrefix('q: ')
        self.qInput.setValue(1)

        self.phiInput = QtWidgets.QSpinBox()
        self.phiInput.setRange(0, 360)
        self.phiInput.setPrefix('φ: ')
        self.phiInput.setSuffix('°')

        thisLayout = QtWidgets.QVBoxLayout()
        thisLayout.setAlignment(QtCore.Qt.AlignTop)
        thisLayout.addWidget(label1)
        thisLayout.addWidget(label2)
        thisLayout.addWidget(self.linkpqCheck)
        thisLayout.addWidget(self.pInput)
        thisLayout.addWidget(self.qInput)
        thisLayout.addWidget(self.phiInput)
        self.setLayout(thisLayout)

        self.linkpqCheck.stateChanged[int].connect(self.link_pq)
        self.pInput.valueChanged.connect(self.calc_x)
        self.qInput.valueChanged.connect(self.calc_y)
        self.phiInput.valueChanged.connect(self.calc_y)

        self.calc_x()
        self.calc_y()

    def link_pq(self, state):
        if state == 2:
            self.qInput.setDisabled(True)
            self.qInput.setValue(self.pInput.value())
        else:
            self.qInput.setDisabled(False)

    def calc_x(self):

        p = self.pInput.value()
        self.x = np.cos(self.t * p)
        if self.linkpqCheck.isChecked():
            self.qInput.setValue(p)
        self.plot()

    def calc_y(self):

        q = self.qInput.value()
        phi = self.phiInput.value() / 180 * np.pi
        self.y = np.cos(self.t * q + phi)
        self.plot()

    def plot(self):

        self.parent.canvasBox.plot(self.x, self.y)


class CanvasBox(QtWidgets.QWidget):
    '''
        Canvas box for plotting time domain and freq domain data
    '''

    def __init__(self, parent):
        ''' Initiate plot canvas '''

        QtWidgets.QWidget.__init__(self)
        self.parent = parent
        # set global pg options
        pg.setConfigOption('leftButtonPan', False)

        canvas1 = pg.PlotWidget(title='Lissajour Curve')
        canvas1.showGrid(x=True, y=True)
        canvas1.setLabel('left')
        canvas1.setLabel('right')
        canvas1.setLabel('top')
        canvas1.setLabel('bottom')
        canvas1.getAxis('top').setStyle(showValues=False)
        canvas1.getAxis('bottom').setStyle(showValues=False)
        canvas1.getAxis('left').setStyle(showValues=False)
        canvas1.getAxis('right').setStyle(showValues=False)
        self.curve1 = canvas1.plot()
        self.curve1.setPen(color='ffb62f', width=1)

        thisLayout = QtWidgets.QVBoxLayout()
        thisLayout.addWidget(canvas1)
        self.setLayout(thisLayout)

    def plot(self, x, y):
        self.curve1.setData(x, y)


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
