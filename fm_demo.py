#! encoding = utf-8

''' Demostrate frequency modulation '''

import sys
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
from scipy.special import wofz

def voigt1(x, sigma, gamma):
    '''
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM sigma centered at 0
    '''

    ss = sigma / np.sqrt(2 * np.log(2))

    return -np.real(wofz((x + 1j*gamma)/ss/np.sqrt(2))) / ss / np.sqrt(2*np.pi)


class MainWindow(QtWidgets.QMainWindow):
    '''
        Implements the main window
    '''

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self)
        self.setStyleSheet('font-size: 10pt; font-family: default')
        self.setWindowTitle('Frequency Modulation Demo')
        self.setMinimumWidth(1200)
        self.setMinimumHeight(800)
        self.resize(QtCore.QSize(1500, 800))
        sGeo = QtWidgets.QDesktopWidget().screenGeometry()
        self.move((sGeo.width()-1500)//2, (sGeo.height()-800)//2)

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
        self.setMaximumWidth(300)
        self.lx = np.linspace(-20, 20, 1001)
        self.t = np.linspace(0, 10, 1001)
        #self.t = np.linspace(0, 2*np.pi, 1000)
        #self.x = np.zeros_like(self.t)
        #self.y = np.zeros_like(self.t)

        labelVoigtDG = QtWidgets.QLabel('Voigt lineshape: Gaussian width')
        labelVoigtDL = QtWidgets.QLabel('Voigt lineshape: Lorentzian width')
        labelModFreq = QtWidgets.QLabel('Modulation frequency')
        labelModDev = QtWidgets.QLabel('Modulation deviation (detuning)')
        labelModDepth = QtWidgets.QLabel('Modulatioon depth (amplitude)')

        self.dgInput = QtWidgets.QDoubleSpinBox()
        self.dgInput.setRange(0.1, 5)
        self.dgInput.setDecimals(1)
        self.dgInput.setSingleStep(0.1)
        self.dgInput.setValue(1)

        self.dlInput = QtWidgets.QDoubleSpinBox()
        self.dlInput.setRange(0, 5)
        self.dlInput.setDecimals(1)
        self.dlInput.setSingleStep(0.1)
        self.dlInput.setValue(0)

        self.modFreqInput = QtWidgets.QDoubleSpinBox()
        self.modFreqInput.setRange(0, 500)
        self.modFreqInput.setDecimals(0)
        self.modFreqInput.setSingleStep(10)
        self.modFreqInput.setValue(100)

        self.modDevInput = QtWidgets.QDoubleSpinBox()
        self.modDevInput.setRange(-10, 10)
        self.modDevInput.setDecimals(1)
        self.modDevInput.setSingleStep(0.1)
        self.modDevInput.setValue(0)

        self.modDepthInput = QtWidgets.QDoubleSpinBox()
        self.modDepthInput.setRange(0, 10)
        self.modDepthInput.setDecimals(1)
        self.modDepthInput.setSingleStep(0.1)
        self.modDepthInput.setValue(1)

        thisLayout = QtWidgets.QFormLayout()
        thisLayout.setAlignment(QtCore.Qt.AlignTop)
        thisLayout.addRow(labelVoigtDG, self.dgInput)
        thisLayout.addRow(labelVoigtDL, self.dlInput)
        thisLayout.addRow(labelModFreq, self.modFreqInput)
        thisLayout.addRow(labelModDev, self.modDevInput)
        thisLayout.addRow(labelModDepth, self.modDepthInput)
        self.setLayout(thisLayout)

        self.dgInput.valueChanged.connect(self.calc_line)
        self.dlInput.valueChanged.connect(self.calc_line)

        self.modFreqInput.valueChanged.connect(self.calc_mod)
        self.modDevInput.valueChanged.connect(self.calc_mod)
        self.modDepthInput.valueChanged.connect(self.calc_mod)

        self.calc_line()
        self.calc_mod()

    def calc_line(self):

        dg = self.dgInput.value()
        dl = self.dlInput.value()
        ly = voigt1(self.lx, dg, dl)
        self.parent.canvasBox.plot_input(self.lx, ly)
        self.calc_mod()

    def calc_mod(self):
        mod_freq = self.modFreqInput.value()
        mod_dev = self.modDevInput.value()
        mod_depth = self.modDepthInput.value()
        dg = self.dgInput.value()
        dl = self.dlInput.value()

        mod_x = mod_depth * np.sin(2*np.pi*mod_freq*self.t*0.01) + mod_dev
        mod_y = voigt1(mod_x, dg, dl)
        self.parent.canvasBox.plot_mod(self.t[0:300], mod_x[0:300])
        self.parent.canvasBox.plot_output(self.t, mod_y)
        self.parent.canvasBox.plot_fft(mod_y, 0.01)



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

        canvasInput = pg.PlotWidget(title='Line')
        canvasInput.showGrid(x=True, y=True)
        canvasInput.setLabel('left')
        canvasInput.setLabel('right')
        canvasInput.setLabel('top')
        canvasInput.setLabel('bottom')
        # Let's display axis values again because the plot region
        # is not garenteed to be square
        #canvas1.getAxis('top').setStyle(showValues=False)
        #canvas1.getAxis('bottom').setStyle(showValues=False)
        #canvas1.getAxis('left').setStyle(showValues=False)
        #canvas1.getAxis('right').setStyle(showValues=False)
        self.curveInput = canvasInput.plot()
        self.curveInput.setPen(color='ffb62f', width=1)
        self.curveMod = pg.PlotCurveItem()
        self.curveMod.setPen(color='32afde', width=2)
        canvasInput.addItem(self.curveMod)

        canvasOutput = pg.PlotWidget(title='Modulation Output')
        canvasOutput.showGrid(x=True, y=True)
        canvasOutput.setLabel('left')
        canvasOutput.setLabel('right')
        canvasOutput.setLabel('top')
        canvasOutput.setLabel('bottom')
        self.curveOutput = canvasOutput.plot()
        self.curveOutput.setPen(color='ffb62f', width=1)
        canvasOutput.setYLink(canvasInput)

        canvasFFT = pg.PlotWidget(title='FFT of Modulation Output')
        canvasFFT.showGrid(x=True, y=True)
        canvasFFT.setLabel('left')
        canvasFFT.setLabel('right')
        canvasFFT.setLabel('top')
        canvasFFT.setLabel('bottom')
        self.curveFFT = canvasFFT.plot()
        self.curveFFT.setPen(color='ffb62f', width=1)

        ## create third ViewBox.
        ## this time we need to create a new axis as well.
        # xxp3 = pg.ViewBox()
        # ax3 = pg.AxisItem('right')
        # canvasInput.layout.addItem(ax3, 2, 3)
        # canvasInput.scene().addItem(p3)
        # ax3.linkToView(p3)
        # p3.setXLink(canvasInput)
        # ax3.setZValue(-10000)
        # ax3.setLabel('axis 3', color='#ff0000')

        thisLayout = QtWidgets.QGridLayout()
        thisLayout.addWidget(canvasInput, 0, 0)
        thisLayout.addWidget(canvasOutput, 0, 1)
        thisLayout.addWidget(canvasFFT, 1, 0, 1, 2)
        self.setLayout(thisLayout)

    def plot_input(self, x, y):
        self.curveInput.setData(x, y)

    def plot_output(self, x, y):
        self.curveOutput.setData(x, y)

    def plot_fft(self, y, fscale):
        x = np.fft.rfftfreq(len(y)) / fscale * 100
        y = np.absolute(np.fft.rfft(y))
        self.curveFFT.setData(x, y)

    def plot_mod(self, t, mod_x):
        ''' Plot modulation sine wave on top of the line (rotate 90 deg) '''

        self.curveMod.setData(mod_x, t*0.1-0.8)

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
