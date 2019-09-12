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
        self.resize(QtCore.QSize(1500, 900))
        sGeo = QtWidgets.QDesktopWidget().screenGeometry()
        self.move((sGeo.width()-1500)//2, (sGeo.height()-900)//2)

        self.canvasBox = CanvasBox(self)
        self.parBox = ParBox(self)

        mainLayout = QtWidgets.QVBoxLayout()
        mainLayout.addWidget(self.parBox)
        mainLayout.addWidget(self.canvasBox)
        # Enable main window
        mainWidget = QtWidgets.QWidget()
        mainWidget.setLayout(mainLayout)
        self.setCentralWidget(mainWidget)



class ParBox(QtWidgets.QGroupBox):

    def __init__(self, parent):
        QtWidgets.QGroupBox.__init__(self)
        self.parent = parent
        self.lx = np.linspace(-20, 20, 1001)
        self.t = np.linspace(0, 10, 1001)
        self.mod_x_array = np.linspace(-10, 10, 201)
        self.mod_x_mat = np.zeros(0)
        self.mod_y_mat = np.zeros(0)
        self.seed = 0

        labelVoigtDG = QtWidgets.QLabel('Voigt lineshape: Gaussian width')
        labelVoigtDL = QtWidgets.QLabel('Voigt lineshape: Lorentzian width')
        labelNoise = QtWidgets.QLabel('Noise level')
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

        self.noiseInput = QtWidgets.QDoubleSpinBox()
        self.noiseInput.setRange(0, 0.5)
        self.noiseInput.setDecimals(2)
        self.noiseInput.setSingleStep(0.01)
        self.noiseInput.setValue(0)

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

        thisLayout = QtWidgets.QGridLayout()
        thisLayout.setAlignment(QtCore.Qt.AlignTop)
        thisLayout.addWidget(labelVoigtDG, 0, 0)
        thisLayout.addWidget(self.dgInput, 0, 1)
        thisLayout.addWidget(labelVoigtDL, 0, 2)
        thisLayout.addWidget(self.dlInput, 0, 3)
        thisLayout.addWidget(labelNoise, 0, 4)
        thisLayout.addWidget(self.noiseInput, 0, 5)
        thisLayout.addWidget(labelModFreq, 1, 0)
        thisLayout.addWidget(self.modFreqInput, 1, 1)
        thisLayout.addWidget(labelModDepth, 1, 2)
        thisLayout.addWidget(self.modDepthInput, 1, 3)
        thisLayout.addWidget(labelModDev, 1, 4)
        thisLayout.addWidget(self.modDevInput, 1, 5)
        self.setLayout(thisLayout)

        self.dgInput.valueChanged.connect(self.calc_line)
        self.dlInput.valueChanged.connect(self.calc_line)
        self.noiseInput.valueChanged.connect(self.calc_line)
        self.modFreqInput.valueChanged.connect(self.calc_fm_spec)
        self.modDepthInput.valueChanged.connect(self.calc_fm_spec)
        self.modDevInput.valueChanged.connect(self.calc_mod)

        self.calc_line()
        self.calc_mod()
        self.calc_fm_spec()

    def calc_line(self):

        dg = self.dgInput.value()
        dl = self.dlInput.value()
        noise = self.noiseInput.value()
        self.seed = int(np.abs(np.random.rand())*2**32)
        np.random.RandomState(seed=self.seed)
        ly = voigt1(self.lx, dg, dl) + np.random.normal(loc=0, scale=noise, size=len(self.lx))
        self.parent.canvasBox.plot_input(self.lx, ly)
        self.calc_fm_spec()

    def calc_fm_spec(self):

        mod_freq = self.modFreqInput.value()
        mod_depth = self.modDepthInput.value()
        noise = self.noiseInput.value()
        dg = self.dgInput.value()
        dl = self.dlInput.value()

        mod_x = mod_depth * np.sin(2*np.pi*mod_freq*self.t*0.01)
        lr = len(mod_x)             # length of row
        lc = len(self.mod_x_array)  # length of column
        self.mod_x_mat = np.repeat(mod_x, lc).reshape((lr, lc)) \
                    + np.repeat(self.mod_x_array, lr).reshape((lc, lr)).transpose()
        np.random.RandomState(seed=self.seed)
        self.mod_y_mat = np.apply_along_axis(voigt1, 1, self.mod_x_mat, dg, dl)\
                         + np.random.normal(loc=0, scale=noise, size=(lr, lc))
        self.fft_y_mat = np.apply_along_axis(np.fft.rfft, 0, self.mod_y_mat)
        # find the 1st, 2nd and 3rd harmonics
        idx_1f = int(mod_freq / 10)
        y0c = self.fft_y_mat[0, :]             # complex
        y1c = self.fft_y_mat[idx_1f, :]        # complex
        y2c = self.fft_y_mat[idx_1f*2, :]      # complex
        y3c = self.fft_y_mat[idx_1f*3, :]      # complex
        # get the absolute value square but keep the sign
        y0 = np.real(y0c)*np.abs(np.real(y0c)) + np.imag(y0c)*np.abs(np.imag(y0c))
        y1 = np.real(y1c)*np.abs(np.real(y1c)) + np.imag(y1c)*np.abs(np.imag(y1c))
        y2 = np.real(y2c)*np.abs(np.real(y2c)) + np.imag(y2c)*np.abs(np.imag(y2c))
        y3 = np.real(y3c)*np.abs(np.real(y3c)) + np.imag(y3c)*np.abs(np.imag(y3c))
        # get the square root and keep the sign
        y0 = -np.sqrt(np.abs(y0)) * np.sign(y0)
        y1 = -np.sqrt(np.abs(y1)) * np.sign(y1)
        y2 = -np.sqrt(np.abs(y2)) * np.sign(y2)
        y3 = -np.sqrt(np.abs(y3)) * np.sign(y3)
        self.parent.canvasBox.plot_fm_spec(self.mod_x_array, y0, y1, y2, y3)

        # plot modulation output signal
        self.calc_mod()

    def calc_mod(self):

        tol = 1e-6
        mod_dev = self.modDevInput.value()
        idx = np.argwhere(np.logical_and(self.mod_x_array<mod_dev+tol, self.mod_x_array>mod_dev-tol))[0,0]
        mod_x = self.mod_x_mat[:, idx]
        mod_y = self.mod_y_mat[:, idx]
        fft_y = self.fft_y_mat[:, idx]
        x = np.fft.rfftfreq(len(mod_y)) * 1e4
        #mod_x = mod_depth * np.sin(2*np.pi*mod_freq*self.t*0.01) + mod_dev
        #mod_y = voigt1(mod_x, dg, dl)
        self.parent.canvasBox.plot_mod(self.t[0:300], mod_x[0:300])
        self.parent.canvasBox.plot_output(self.t, mod_y)
        self.parent.canvasBox.plot_fft(x, fft_y)


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

        canvasInput = pg.PlotWidget(title='Spectral Line')
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
        canvasFFT.addLegend(offset=(0,0))
        self.curveFFTre = canvasFFT.plot(name='real')
        self.curveFFTre.setPen(color='ffb62f', width=1)
        self.curveFFTim = pg.PlotCurveItem(name='imag')
        self.curveFFTim.setPen(color='32afde', width=1)
        self.curveFFTamp = pg.PlotCurveItem(name='amp')
        self.curveFFTamp.setPen(color='e0e0e0', width=1)
        canvasFFT.addItem(self.curveFFTim)
        canvasFFT.addItem(self.curveFFTamp)

        canvasFM = pg.PlotWidget(title='FM spectrum (Opposite phase)')
        canvasFM.showGrid(x=True, y=True)
        canvasFM.setLabel('left')
        canvasFM.setLabel('right')
        canvasFM.setLabel('top')
        canvasFM.setLabel('bottom')
        canvasFM.addLegend(offset=(0,0))
        self.curveFM0f = canvasFM.plot(name='DC(×0.5)')
        self.curveFM0f.setPen(color='e0e0e0', width=1, style=QtCore.Qt.DashLine)
        self.curveFM1f = pg.PlotCurveItem(name='1st harmonic')
        self.curveFM1f.setPen(color='ffb62f', width=1)
        self.curveFM2f = pg.PlotCurveItem(name='2nd harmonic')
        #self.curveFM2f.plotItem.legend.addItem('2nd harmonic')
        self.curveFM2f.setPen(color='8edeb3', width=1)
        self.curveFM3f = pg.PlotCurveItem(name='3rd harmonic')
        #self.curveFM3f.plotItem.legend.addItem('3rd harmonic')
        self.curveFM3f.setPen(color='32afde', width=1)
        canvasFM.addItem(self.curveFM1f)
        canvasFM.addItem(self.curveFM2f)
        canvasFM.addItem(self.curveFM3f)
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
        thisLayout.addWidget(canvasFM, 1, 0)
        thisLayout.addWidget(canvasFFT, 1, 1)
        self.setLayout(thisLayout)

    def plot_input(self, x, y):
        self.curveInput.setData(x, y)

    def plot_output(self, x, y):
        self.curveOutput.setData(x, y)

    def plot_fft(self, x, y):

        self.curveFFTre.setData(x, np.real(y))
        self.curveFFTim.setData(x, np.imag(y))
        self.curveFFTamp.setData(x, np.absolute(y))

    def plot_mod(self, t, mod_x):
        ''' Plot modulation sine wave on top of the line (rotate 90 deg) '''

        self.curveMod.setData(mod_x, t*0.1-0.8)

    def plot_fm_spec(self, x, y0, y1, y2, y3):

        self.curveFM0f.setData(x, y0*0.5)
        self.curveFM1f.setData(x, y1)
        self.curveFM2f.setData(x, y2)
        self.curveFM3f.setData(x, y3)


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
