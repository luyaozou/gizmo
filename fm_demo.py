#! encoding = utf-8

''' Demostrate frequency modulation '''

import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from scipy.special import wofz
from scipy.signal import square

def voigt(x, gg, ll, der, x0=0, y0=0):
    """ Return the Voigt line shape at x with Lorentzian component FWHM ll
    and Gaussian component FWHM gg

    Voigt profile defined by recursive relations (n>=0).
    voigtnd = Re[D_n(w)] / (2*sigma^2*sqrt(pi))
    :arguments
        x: float / np1darray    x
        gg: float               Gaussian FWHM. gg = 2*sqrt(2ln2)*sigma
        ll: float               Lorentzian FWMH. ll = 2*gamma
        x0: float               center
        der: int                #-th derivative
        y0: float               y shift y'=y+y0 (used for root finding)

    :returns
        vnd: float/np1darray    the profile
    """

    # calculate Gaussian stdev and Lorentzian gamma
    sigma = gg / (2 * np.sqrt(2 * np.log(2)))
    gamma = ll / 2
    # the complex z for the Faddeeva function
    z = (x - x0 + 1j*gamma) / (sigma*np.sqrt(2))
    return -np.real(_wofznd(z, der)) + y0

def _wofznd(z, n):
    """
    Return the n-th derivative of the Faddeeva function wofz(z).
    The equation comes from Heinzel, P., Astronomical Institutes of
    Czechoslovakia, Bulletin, vol. 29, no. 3, 1978, p. 159-162.

    The derivative is defined by recursive relations:
    D[w] = e^(w^2)*erfc(w),  w = a-1j*u
    D1[w] = 2*w*D[w] - 2/sqrt(pi)
    D_n[w] = 2*w*D_(n-1)[w] + 2*(n-1)D_(n-2)[w] (n>=2)

    w[z] = exp(-z**2)*erfc(-1j*z)
    w_1[z] = -2z * w[z] + 2j/sqrt(pi)
    w_n[z] = -2*z*w_(n-1)[z] - 2*(n-1)*w_(n-2)[w] (n>=2)

    where a and u are related to Gaussian and Lorentzian components by
    - Gaussian(sigma) = exp(-u^2),  u^2=x^2/(2*sigma)
    - Lorentzian(gamma) = u/pi*(u^2+a^2),  a=gamma/(sqrt(2)*sigma)

    The link between voigt and D[w] is
    - voigt = Re[D(w)] / (2*sigma^2*sqrt(pi))

    :arguments
        w: complex              The complex variable w=a-1j*u
        n: int                  n-th derivative
    :returns
        w_n[z]: complex / np1darray
    """

    if n < 0:
        raise ValueError('n must >= 0')
    elif n == 0:
        return wofz(z)
    elif n == 1:
        return -2*z*wofz(z) + 2j/np.sqrt(np.pi)
    else:
        return -2*z*_wofznd(z, n-1) - 2*(n-1)*_wofznd(z, n-2)


def fwhm_voi(gg, ll):
    """ Approximate voigt FWHM calculated from analytical expressions
    Olivero (1977) JQSRT 17, 233-236, medium accurate version
    :arguments
        gg: float       FWHM of Gaussian component
        ll: float       FWHM of Lorentzian component
        approx: str     approximation accuracy level
    :returns
        vv: float       FWHM
    """

    return 0.5346 * ll + np.sqrt(0.2166 * ll**2 + gg**2)


def triangle(x, f):

    period = 1./f
    # number of points for each period
    n = round(period / (x[1] - x[0]))
    rising = np.linspace(0, 2, n // 2)
    falling = rising[::-1][1:-1]
    wave_1t = np.concatenate((rising, falling))
    n1t = len(wave_1t)
    wave = np.zeros((len(x) // n1t+1) * n1t)
    # number of iteration
    for i in range(len(x) // n1t + 1):
        wave[i * n1t: (i+1) * n1t] = wave_1t
    return wave[:len(x)] - 1


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

        labelVoigtDG = QtWidgets.QLabel('Voigt lineshape: Gaussian FWHM')
        labelVoigtDL = QtWidgets.QLabel('Voigt lineshape: Lorentzian FWHM')
        labelNoise = QtWidgets.QLabel('Noise Level')
        labelModFreq = QtWidgets.QLabel('Modulation Frequency')
        labelModDev = QtWidgets.QLabel('Detuning')
        labelModDepth = QtWidgets.QLabel('Modulation Depth (amplitude)')
        labelWaveform = QtWidgets.QLabel('Modulation Waveform')
        labelDetHarm = QtWidgets.QLabel('Detecting Harmonic')

        self.dgInput = QtWidgets.QDoubleSpinBox()
        self.dgInput.setRange(0.1, 5)
        self.dgInput.setDecimals(1)
        self.dgInput.setSingleStep(0.1)
        self.dgInput.setValue(1)

        self.dlInput = QtWidgets.QDoubleSpinBox()
        self.dlInput.setRange(0, 5)
        self.dlInput.setDecimals(1)
        self.dlInput.setSingleStep(0.1)
        self.dlInput.setValue(1)

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
        self.modDepthInput.setDecimals(2)
        self.modDepthInput.setStepType(1)
        self.modDepthInput.setValue(1)
        self.modDepthInput.setSuffix(' ×FWHM')

        self.comboWaveform = QtWidgets.QComboBox()
        self.comboWaveform.addItems(['Sine', 'Square', 'Triangle'])

        self.harmInput = QtWidgets.QSpinBox()
        self.harmInput.setMinimum(0)

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
        thisLayout.addWidget(labelWaveform, 2, 0)
        thisLayout.addWidget(self.comboWaveform, 2, 1)
        thisLayout.addWidget(labelDetHarm, 2, 4)
        thisLayout.addWidget(self.harmInput, 2, 5)
        self.setLayout(thisLayout)

        self.dgInput.valueChanged.connect(self.calc_line)
        self.dlInput.valueChanged.connect(self.calc_line)
        self.noiseInput.valueChanged.connect(self.calc_line)
        self.modFreqInput.valueChanged.connect(self.calc_fm_spec)
        self.modDepthInput.valueChanged.connect(self.calc_fm_spec)
        self.modDevInput.valueChanged.connect(self.calc_mod)
        self.comboWaveform.currentIndexChanged.connect(self.calc_fm_spec)
        self.harmInput.valueChanged.connect(self.calc_fm_spec)

        self.calc_line()
        self.calc_mod()
        self.calc_fm_spec()

    def calc_line(self):

        dg = self.dgInput.value()
        dl = self.dlInput.value()
        noise = self.noiseInput.value()
        self.seed = int(np.abs(np.random.rand())*2**32)
        np.random.RandomState(seed=self.seed)
        ly = voigt(self.lx, dg, dl, 0) + np.random.normal(loc=0, scale=noise, size=len(self.lx))
        self.parent.canvasBox.plot_input(self.lx, ly)
        self.calc_fm_spec()

    def calc_fm_spec(self):

        dg = self.dgInput.value()
        dl = self.dlInput.value()
        wf = self.comboWaveform.currentText()
        der = self.harmInput.value()
        mod_freq = self.modFreqInput.value()
        mod_depth = self.modDepthInput.value() * fwhm_voi(dg, dl)
        noise = self.noiseInput.value()

        if wf == 'Sine':
            mod_x = 0.5 * mod_depth * np.sin(2*np.pi*mod_freq*self.t*0.01)
        elif wf == 'Square':
            mod_x = 0.5 * mod_depth * square(2*np.pi*mod_freq*self.t*0.01)
        elif wf == 'Triangle':
            mod_x = 0.5 * mod_depth * triangle(self.t*0.01, mod_freq)
        else:
            mod_x = np.ones(len(self.t))

        lr = len(mod_x)             # length of row
        lc = len(self.mod_x_array)  # length of column
        self.mod_x_mat = np.repeat(mod_x, lc).reshape((lr, lc)) \
                    + np.repeat(self.mod_x_array, lr).reshape((lc, lr)).transpose()
        np.random.RandomState(seed=self.seed)
        self.mod_y_mat = np.apply_along_axis(voigt, 1, self.mod_x_mat, dg, dl, 0)\
                         + np.random.normal(loc=0, scale=noise, size=(lr, lc))
        self.fft_y_mat = np.apply_along_axis(np.fft.rfft, 0, self.mod_y_mat)

        # find the harmonics component
        idx_1f = int(mod_freq / 10)
        ync = self.fft_y_mat[idx_1f * der, :]   # complex
        #y0c = self.fft_y_mat[0, :]             # complex
        #y1c = self.fft_y_mat[idx_1f, :]        # complex
        #y2c = self.fft_y_mat[idx_1f*2, :]      # complex
        #y3c = self.fft_y_mat[idx_1f*3, :]      # complex
        # this step is required to get the correct line shape
        # get the absolute value square but keep the sign
        yn = np.real(ync)*np.abs(np.real(ync)) + np.imag(ync)*np.abs(np.imag(ync))
        # y0 = np.real(y0c)*np.abs(np.real(y0c)) + np.imag(y0c)*np.abs(np.imag(y0c))
        # y1 = np.real(y1c)*np.abs(np.real(y1c)) + np.imag(y1c)*np.abs(np.imag(y1c))
        # y2 = np.real(y2c)*np.abs(np.real(y2c)) + np.imag(y2c)*np.abs(np.imag(y2c))
        # y3 = np.real(y3c)*np.abs(np.real(y3c)) + np.imag(y3c)*np.abs(np.imag(y3c))
        # get the square root and keep the sign
        yn = np.sqrt(np.abs(yn)) * np.sign(yn)
        # y0 = -np.sqrt(np.abs(y0)) * np.sign(y0)
        # y1 = -np.sqrt(np.abs(y1)) * np.sign(y1)
        # y2 = -np.sqrt(np.abs(y2)) * np.sign(y2)
        # y3 = -np.sqrt(np.abs(y3)) * np.sign(y3)

        # calculate theoretical voigt profile
        yv = voigt(self.mod_x_array, dg, dl, der)
        # normalize the voigt profile to the same intensity as yn
        yv = yv / yv.ptp() * yn.ptp()
        # put the yn as the same sign as yv
        # we integrate the difference of the two profiles and see which one is
        # smaller. the smaller one has the same sign
        itg1 = np.sum(np.abs(yn - yv))
        itg2 = np.sum(np.abs(yn + yv))
        if itg1 > itg2:
            yv = -yv
        else:
            pass
        self.parent.canvasBox.plot_fm_spec(self.mod_x_array, yn, yv)

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
        self.curveInput.setPen(pg.mkPen(color='#ffb62f', width=1))
        self.curveMod = pg.PlotCurveItem()
        self.curveMod.setPen(pg.mkPen(color='#32afde', width=2))
        canvasInput.addItem(self.curveMod)

        canvasOutput = pg.PlotWidget(title='Modulation Output')
        canvasOutput.showGrid(x=True, y=True)
        canvasOutput.setLabel('left')
        canvasOutput.setLabel('right')
        canvasOutput.setLabel('top')
        canvasOutput.setLabel('bottom')
        self.curveOutput = canvasOutput.plot()
        self.curveOutput.setPen(pg.mkPen(color='#ffb62f', width=1))
        canvasOutput.setYLink(canvasInput)

        canvasFFT = pg.PlotWidget(title='FFT of Modulation Output')
        canvasFFT.showGrid(x=True, y=True)
        canvasFFT.setLabel('left')
        canvasFFT.setLabel('right')
        canvasFFT.setLabel('top')
        canvasFFT.setLabel('bottom')
        canvasFFT.addLegend(offset=(0,0))
        self.curveFFTre = canvasFFT.plot(name='real')
        self.curveFFTre.setPen(pg.mkPen(color='#ffb62f', width=1))
        self.curveFFTim = pg.PlotCurveItem(name='imag')
        self.curveFFTim.setPen(pg.mkPen(color='#32afde', width=1))
        self.curveFFTamp = pg.PlotCurveItem(name='amp')
        self.curveFFTamp.setPen(pg.mkPen(color='#e0e0e0', width=1))
        canvasFFT.addItem(self.curveFFTim)
        canvasFFT.addItem(self.curveFFTamp)

        canvasFM = pg.PlotWidget(title='FM spectrum (Opposite phase)')
        canvasFM.showGrid(x=True, y=True)
        canvasFM.setLabel('left')
        canvasFM.setLabel('right')
        canvasFM.setLabel('top')
        canvasFM.setLabel('bottom')
        canvasFM.addLegend(offset=(0,0))
        self.curveFMnf = canvasFM.plot(name='FM spectrum')
        self.curveFMnf.setPen(pg.mkPen(color='#ffb62f', width=1))
        self.curveVdn = pg.PlotCurveItem(name='Voigt derivative')
        self.curveVdn.setPen(pg.mkPen(color='#e0e0e0', width=1, style=QtCore.Qt.DashLine))
        # self.curveFM0f = canvasFM.plot(name='DC(×0.5)')
        # self.curveFM0f.setPen(pg.mkPen(color='#e0e0e0', width=1, style=QtCore.Qt.DashLine))
        # self.curveFM1f = pg.PlotCurveItem(name='1st harmonic')
        # self.curveFM1f.setPen(pg.mkPen(color='#ffb62f', width=1))
        # self.curveFM2f = pg.PlotCurveItem(name='2nd harmonic')
        # #self.curveFM2f.plotItem.legend.addItem('2nd harmonic')
        # self.curveFM2f.setPen(pg.mkPen(color='#8edeb3', width=1))
        # self.curveFM3f = pg.PlotCurveItem(name='3rd harmonic')
        # #self.curveFM3f.plotItem.legend.addItem('3rd harmonic')
        # self.curveFM3f.setPen(pg.mkPen(color='#32afde', width=1))
        # canvasFM.addItem(self.curveFM1f)
        # canvasFM.addItem(self.curveFM2f)
        # canvasFM.addItem(self.curveFM3f)
        canvasFM.addItem(self.curveVdn)
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
        thisLayout.addWidget(canvasOutput, 1, 0)
        thisLayout.addWidget(canvasFM, 0, 1)
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

    def plot_fm_spec(self, x, yn, yv):
        """
        y: actual data
        yv: theoretical voigt profile
        """
        self.curveFMnf.setData(x, yn)
        self.curveVdn.setData(x, yv)
        #   self.curveFM0f.setData(x, y0*0.5)
        #   self.curveFM1f.setData(x, y1)
        #   self.curveFM2f.setData(x, y2)
        #   self.curveFM3f.setData(x, y3)


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
