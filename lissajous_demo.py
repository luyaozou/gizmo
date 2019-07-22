#! encoding = utf-8

''' Demostrate lissajous curve '''

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
        self.setWindowTitle('Lissajous Curve Demo')

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

        label1 = QtWidgets.QLabel('x = A * cos(p*t)')
        label1.setStyleSheet('font-size: 14pt')
        label2 = QtWidgets.QLabel('y = B * cos(q*t + φ)')
        label2.setStyleSheet('font-size: 14pt')

        self.linkpqCheck = QtWidgets.QCheckBox('Link p, q')
        self.linkABcheck = QtWidgets.QCheckBox('Link A, B')

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

        self.aInput = QtWidgets.QDoubleSpinBox()
        self.aInput.setPrefix('A: ')
        self.aInput.setValue(1)
        self.aInput.setStepType(1)
        self.aInput.setDecimals(3)
        self.aInput.setRange(1e-3, 1e3)

        self.bInput = QtWidgets.QDoubleSpinBox()
        self.bInput.setPrefix('B: ')
        self.bInput.setValue(1)
        self.bInput.setStepType(1)
        self.bInput.setDecimals(3)
        self.bInput.setRange(1e-3, 1e3)

        self.openBtn = QtWidgets.QPushButton('Open data file')
        self.openBtn.clicked.connect(self.open_file)
        self.clearBtn = QtWidgets.QPushButton('Clear data')
        self.clearBtn.clicked.connect(self.clear_data)

        thisLayout = QtWidgets.QVBoxLayout()
        thisLayout.setAlignment(QtCore.Qt.AlignTop)
        thisLayout.addWidget(label1)
        thisLayout.addWidget(label2)
        thisLayout.addWidget(self.linkpqCheck)
        thisLayout.addWidget(self.linkABcheck)
        thisLayout.addWidget(self.pInput)
        thisLayout.addWidget(self.qInput)
        thisLayout.addWidget(self.phiInput)
        thisLayout.addWidget(self.aInput)
        thisLayout.addWidget(self.bInput)
        thisLayout.addWidget(self.openBtn)
        thisLayout.addWidget(self.clearBtn)
        self.setLayout(thisLayout)

        self.linkpqCheck.stateChanged[int].connect(self.link_pq)
        self.linkABcheck.stateChanged[int].connect(self.link_ab)
        self.pInput.valueChanged.connect(self.calc_x)
        self.qInput.valueChanged.connect(self.calc_y)
        self.phiInput.valueChanged.connect(self.calc_y)
        self.aInput.valueChanged.connect(self.calc_x)
        self.bInput.valueChanged.connect(self.calc_y)

        self.calc_x()
        self.calc_y()

    def link_pq(self, state):
        if state == 2:
            self.qInput.setDisabled(True)
            self.qInput.setValue(self.pInput.value())
        else:
            self.qInput.setDisabled(False)

    def link_ab(self, state):
        if state == 2:
            self.bInput.setDisabled(True)
            self.bInput.setValue(self.aInput.value())
        else:
            self.bInput.setDisabled(False)

    def calc_x(self):

        p = self.pInput.value()
        a = self.aInput.value()
        self.x = np.cos(self.t * p) * a
        if self.linkpqCheck.isChecked():
            self.qInput.setValue(p)
            self.bInput.setValue(a)
        self.plot()

    def calc_y(self):

        q = self.qInput.value()
        b = self.bInput.value()
        phi = self.phiInput.value() / 180 * np.pi
        self.y = np.cos(self.t * q + phi) * b
        self.plot()

    def plot(self):

        self.parent.canvasBox.plot(self.x, self.y)

    def open_file(self):

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self,
            'Open Data Files', '.', 'Time domain data 2 columns (*.*)')

        if not filename:
            pass
        else:
            try:
                data = np.loadtxt(filename)
            except DataError:
                d = QtWidgets.QMessageBox(QtGui.QMessageBox.Warning, 'Incorrect data format', '')
                d.exec_()

            self.parent.canvasBox.plot_scatter(data[:, 0]*1e3, data[:, 1]*1e3)

    def clear_data(self):

        self.parent.canvasBox.clear_scatter()


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

        canvas1 = pg.PlotWidget()
        canvas1.showGrid(x=True, y=True)
        canvas1.setLabel('left')
        canvas1.setLabel('right')
        canvas1.setLabel('top')
        canvas1.setLabel('bottom')
        # Let's display axis values again because the plot region
        # is not garenteed to be square
        #canvas1.getAxis('top').setStyle(showValues=False)
        #canvas1.getAxis('bottom').setStyle(showValues=False)
        #canvas1.getAxis('left').setStyle(showValues=False)
        #canvas1.getAxis('right').setStyle(showValues=False)
        self.curve1 = canvas1.plot()
        self.curve1.setPen(color='ffb62f', width=1)
        self.scatter = pg.ScatterPlotItem()
        self.scatter.setPen(color='c0c0c0')
        self.scatter.setSize(1)
        canvas1.addItem(self.scatter)

        thisLayout = QtWidgets.QVBoxLayout()
        thisLayout.addWidget(canvas1)
        self.setLayout(thisLayout)

    def plot(self, x, y):
        self.curve1.setData(x, y)

    def plot_scatter(self, x, y):

        self.scatter.setData(x, y)

    def clear_scatter(self):

        self.scatter.setData(np.zeros(0))


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
