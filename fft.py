#! encoding = utf-8

''' This script is used for performing basic fft transformation
and filtering for the PhLAM mm chirped pulse spectrum.
It also provides least-square fit (uses lmfit) for the processed spectrum.
'''

import sys
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import lmfit
from scipy.special import wofz

DEFAULT_DIR = '/home/luyao/Documents/Data'
INTVL = 50  # waiting interval for batch process
ABS_RANGE = 100 # maximum absorption freq range +/- full FFT freq range (MHz)


def gaussian(x, sigma):
    ''' Return Gaussian line shape at x with HWHM sigma centered at 0'''

    return np.sqrt(np.log(2)/np.pi) / sigma * np.exp(-(x/sigma)**2 * np.log(2))

def lorentzian(x, gamma):
    ''' Return Lorentzian line shape at x with HWHM gamma centered at 0'''

    return gamma / np.pi / (x**2 + gamma**2)

def voigt1(x, sigma, gamma):
    '''
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM sigma centered at 0
    '''

    ss = sigma / np.sqrt(2 * np.log(2))

    return np.real(wofz((x + 1j*gamma)/ss/np.sqrt(2))) / ss / np.sqrt(2*np.pi)


def f2min(pars, x, y):
    ''' Voigt function to be minimized.
    Arguments:
        pars: parameter list = [x0#, a#, sigma#, gamma#]
        x: np.array
        y: np.array
    Returns:
        res: np.array = f(x)-y
    '''

    v = pars.valuesdict()
    model = 0
    for n in range(len(v)//4):
        model += voigt1(x - v['x0'+str(n)], v['sigma'+str(n)], v['gamma'+str(n)]) * v['a'+str(n)]

    return model - y


def peak_search(x, y, w):
    ''' Peak search algorithm.
    Arguments:
        x: np.array    x data
        y: np.array    y data
        w: width       int
    Returns:
        [[x0, a0], ]: np.array of initial guesses for x0 and a0
    '''

    # test algorithm
    peak = np.argmax(y)

    return np.column_stack((x[peak], y[peak]*1.5))


class MainWindow(QtWidgets.QMainWindow):
    '''
        Implements the main window
    '''
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self)
        self.setStyleSheet('font-size: 10pt; font-family: default')

        # Set global window properties
        self.setWindowTitle('Chirped Pulse FFT')
        self.setMinimumWidth(800)
        self.setMinimumHeight(800)
        self.resize(QtCore.QSize(1200, 850))

        # initiate component widgets
        self._init_menubar()
        self._init_canvas()
        self._init_parbox()
        self.tdsData = TDSData()
        self.absData = ABSData()

        # Set window layout
        self.mainLayout = QtWidgets.QHBoxLayout()
        self.mainLayout.setSpacing(6)
        self.mainLayout.addWidget(self.canvasBox)
        self.mainLayout.addWidget(self.parWidget)

        # Enable main window
        self.mainWidget = QtWidgets.QWidget()
        self.mainWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.mainWidget)

    def _init_menubar(self):
        ''' Initiate menu bar '''

        # fft part
        self.openFileAction = QtGui.QAction('Open File', self)
        self.openFileAction.setShortcut('Ctrl+O')
        self.openFileAction.triggered.connect(self._open_file)
        self.batchAction = QtGui.QAction('Batch Process', self)
        self.batchAction.setShortcut('Ctrl+Shift+B')
        self.batchAction.triggered.connect(self._batch)
        #self.refFileAction = QtGui.QAction('Open Absorption Ref', self)
        #elf.refFileAction.setShortcut('Ctrl+Shift+R')
        #self.refFileAction.triggered.connect(self._open_ref_file)
        self.saveFileAction = QtGui.QAction('Save File', self)
        self.saveFileAction.setShortcut('Ctrl+S')
        self.saveFileAction.triggered.connect(self._save)
        self.statusBar()

        menuFile = self.menuBar().addMenu('&File')
        menuFile.addAction(self.openFileAction)
        menuFile.addAction(self.batchAction)
        #menuFile.addAction(self.refFileAction)
        menuFile.addAction(self.saveFileAction)

        # fit file part
        self.openFitDiagAction = QtGui.QAction('Open Fit Window', self)
        self.openFitDiagAction.setShortcut('Ctrl+Shift+F')
        self.openFitDiagAction.triggered.connect(self._open_fit_diag)
        fitFile = self.menuBar().addMenu('&Fit')
        fitFile.addAction(self.openFitDiagAction)
        self.fitDiag = FitDiag(self)

    def _init_canvas(self):
        ''' Initiate plot canvas '''

        # for time domain spectrum
        tdsCanvas = pg.PlotWidget(title='Time domain spectrum')
        tdsCanvas.setLabel('left', text='Voltage', units='V')
        tdsCanvas.setLabel('right')
        tdsCanvas.setLabel('bottom', text='Time', units='s')
        tdsCanvas.showGrid(x=True, y=True, alpha=0.8)
        self.tdsCurve = tdsCanvas.plot()
        self.tdsCurve.setPen(color='w', width=1)
        # for window function mask
        self.winFCurve = pg.PlotCurveItem()
        tdsCanvas.addItem(self.winFCurve)
        self.winFCurve.setPen(color='ff9b89', width=1)
        self.winFCurve.setBrush(color=pg.hsvColor(0.09, 0.46, 1, 0.5))
        self.winFCurve.setFillLevel(0)

        # for freq domain spectrum
        fdsCanvas = pg.PlotWidget(title='Frequency domain spectrum')
        fdsCanvas.setLabel('left', text='Intensity', units='')
        fdsCanvas.setLabel('right')
        fdsCanvas.setLabel('bottom', text='FFT Frequency', units='Hz')
        fdsCanvas.showGrid(x=True, y=True, alpha=0.8)
        fdsCanvas.invertX(True)
        self.fdsCurve = fdsCanvas.plot()
        self.fdsCurve.setPen(color='FFB62F', width=1.5)

        # for absorption ref spectrum
        # absCanvas = pg.PlotWidget(title='Absorption spectrum reference')
        # absCanvas.setLabel('left', text='Intensity', units='a.u.')
        # absCanvas.setLabel('bottom', text='Line Frequency', units='Hz')
        # absCanvas.showGrid(x=True, y=True, alpha=0.8)
        # # fdsCanvas.setXLink(absCanvas)   # link two views
        # self.absCurve = absCanvas.plot()
        # self.absCurve.setPen(color='5dcfe2', width=1.5)
        # absCanvas.setXLink(fdsCanvas)

        self.canvasBox = QtWidgets.QWidget()
        canvasLayout = QtWidgets.QVBoxLayout()  # layout for canvas
        canvasLayout.addWidget(tdsCanvas)  # time domain spectrum canvas
        canvasLayout.addWidget(fdsCanvas)  # freq domain spectrum canvas
        #canvasLayout.addWidget(absCanvas)  # absorption spectrum canvas
        self.canvasBox.setLayout(canvasLayout)

    def _init_parbox(self):
        ''' initiate parameter box widgets '''

        self.parWidget = QtWidgets.QWidget()
        parLayout = QtWidgets.QGridLayout()    # layout for parameters
        self.infoBox = InfoBox(self)    # for scan info (from file header)
        self.fftBox = FFTBox(self)      # for fft window & other settings
        self.filterBox = FilterBox(self)    # for filter settings
        self.calcBtn = QtWidgets.QPushButton('Calc')
        self.calcBtn.clicked.connect(self.calc)
        self.saveBtn = QtWidgets.QPushButton('Save')
        self.saveBtn.clicked.connect(self._save)
        self.batchBtn = QtWidgets.QPushButton('Batch')
        self.batchBtn.clicked.connect(self._batch)
        # disable the button unless data file is loaded
        self.calcBtn.setDisabled(True)
        self.saveBtn.setDisabled(True)
        self.batchBtn.setDisabled(True)

        # add widgets & set up layout
        parLayout.addWidget(self.infoBox, 0, 0, 1, 2)
        parLayout.addWidget(self.fftBox, 1, 0, 1, 2)
        parLayout.addWidget(self.filterBox, 2, 0, 1, 2)
        parLayout.addWidget(self.calcBtn, 3, 0, 1, 1)
        parLayout.addWidget(self.saveBtn, 3, 1, 1, 1)
        parLayout.addWidget(self.batchBtn, 4, 0, 1, 2)
        self.parWidget.setLayout(parLayout)

    def wfPlot(self):
        ''' Plot window function on top of the time domain spectrum '''

        if self.tdsData.isData:
            wf = self.filterBox.getWinF(self.tdsData.acqN + 1)
            # adjust wf maximum to the maximum of data & concatenate x
            x = self.tdsData.tdsSpec[:, 0]
            y = self.tdsData.tdsSpec[:, 1]
            self.winFCurve.setData(x, wf*np.max(y))
        else:
            pass

    def fdsPlot(self):
        ''' Plot frequency domain spectrum.
            Public available to receive signals from other widgets.
        '''

        # restrict the spectrum frequency to the expected chirp frequency
        if self.fftBox.limitFreqCheck.isChecked():
            self.fdsCurve.setData(self.fdsSpecLimit)
        else:   # full scale
            self.fdsCurve.setData(self.fdsSpecFull)

    def _open_file(self):
        ''' Open a single data file '''

        # open file dialog
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                'Open Data File', self.tdsData.tdsFileDir, 'Time domain spectrum (*.tdf)')
        # load data file
        status = self.tdsData.load_file(filename)
        if status:  # sucessfully load file
            # refresh info box
            self.infoBox.refresh()
            # plot data on tds canvas
            self.tdsCurve.setData(self.tdsData.tdsSpec)
            self.wfPlot()
            # enable buttons and panels
            self.fftBox.setDisabled(False)
            self.filterBox.setDisabled(False)
            self.calcBtn.setDisabled(False)
            self.saveBtn.setDisabled(False)
            self.batchBtn.setDisabled(False)
            # update lower & upper limits for the fft window
            self.fftBox.setInitInput()
            # initiate one fft plot
            self.calc()
        else:
            pass

    def _open_fit_diag(self):
        ''' Open spectral fit dialog '''

        self.fitDiag.exec_()

    def _open_ref_file(self):
        ''' Open an absorption spectrum as a reference '''

        # disable this when no fds spectrum
        if self.tdsData.isData:
            # open file dialog
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                'Open Data File', '/home/luyao/Documents/Data', 'Absorption Spectrum (*.txt, *.csv, *.dat)')
            # load data file (provide fft freq range)
            status = self.absData.load_file(filename,
                        self.tdsData.minFreq, self.tdsData.maxFreq)
            if status:
                # adjust position shift
                self.absCurve.setPos(-self.tdsData.detFreq, 0)
                self.absCurve.setData(self.absData.absSpec)
            else:
                pass
        else:
            d = QtWidgets.QMessageBox(QtGui.QMessageBox.Warning, 'No time domain data', 'Please load valid time domain data before loading reference spectrum.')
            d.exec_()

    def calc(self, wfname='None'):
        ''' Calculate fft with selected window function '''

        # apply window function to the full tds spectrum before truncating
        wf = self.filterBox.getWinF(self.tdsData.acqN + 1)
        y = self.tdsData.tdsSpec[:, 1] * wf

        # check the validity of i_min and i_max
        bool, i_min, i_max = check_fft_range(self.fftBox.fftMin(),
            self.fftBox.fftMax(), self.tdsData.acqN)
        if bool:
            # restore normal text color
            self.fftBox.fftMinInput.setStyleSheet('color: black')
            self.fftBox.fftMaxInput.setStyleSheet('color: black')
            y = y[i_min:i_max+1]    # account for the python way of indexing
        else:   # invalid i_min & i_max will not take any affect
            # set warning text color
            self.fftBox.fftMinInput.setStyleSheet('color: #D63333')
            self.fftBox.fftMaxInput.setStyleSheet('color: #D63333')

        # add zero-padding
        zp = self.fftBox.zeroPadding()
        if zp:
            yz = np.zeros(len(y) * zp)
            y = np.concatenate((y, yz))
        else:
            pass

        # fft
        fft_y = np.fft.rfft(y)
        # calculate corresponding frequency
        f = np.fft.rfftfreq(len(y)) * self.tdsData.adcCLK

        # chop off the 0 frequency & concatenate spectrum
        self.fdsSpecFull = np.column_stack((f[1:], np.absolute(fft_y[1:])))

        # calculate chirp frequency range & cut-low frequency range
        fmin = self.tdsData.imFreq * 1e6
        fmax = (self.tdsData.imFreq + self.tdsData.spanFreq) * 1e6
        fcut = self.fftBox.cutLowFreq()
        idx = np.logical_and(f[1:] >= max(fmin, fcut), f[1:] <= fmax)
        self.fdsSpecLimit = self.fdsSpecFull[idx, :]

        # flip the array to get increasing frequency
        self.fdsSpecFull = np.flipud(self.fdsSpecFull)
        self.fdsSpecLimit = np.flipud(self.fdsSpecLimit)
        # plot spectrum
        self.fdsPlot()

    def _save(self):
        ''' Save spectrum '''

        if self.tdsData.isData:
            if self.fftBox.limitFreqCheck.isChecked():
                spec = self.fdsSpecLimit
            else:
                spec = self.fdsSpecFull

            # rescale the frequency to MHz unit & adjust to line frequency
            spec[:, 0] = self.tdsData.detFreq - spec[:, 0] * 1e-6
            # prepare header
            hd = self.getHeader()

            # by default, save file to the same directory of the tds file
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(self,
                    'Save Spectrum', self.tdsData.tdsFileDir, 'Frequency domain spectrum (*)')
            if filename:
                np.savetxt(filename, spec, delimiter='\t',
                           fmt=['%.4f', '%8e'], header=hd)
            else:
                pass
        else:
            d = QtWidgets.QMessageBox(QtGui.QMessageBox.Warning, 'No time domain data', 'Please analyze the data before saving.')
            d.exec_()

    def _batch(self):
        ''' Batch Process '''

        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(self,
            'Open Data Files', self.tdsData.tdsFileDir, 'Time domain spectrum (*.tdf)')

        if filenames:
            progD = BatchProc(self, filenames, type='fft')
            progD.exec_()
        else:
            pass

    def getHeader(self):
        ''' generate header information '''

        # prelogue
        hd = 'This spectrum is generated by fft.py (Luyao Zou).\n'
        # scan info
        hd += '{:>12s} | {:>6s} | {:>6s} | {:>6s} | {:>6s} \n'.format(
                'f_start', 'f_span', 'f_im', 'f_clk', 'AcqN')
        hd += '{:12.4f} | {:6.2f} | {:6.2f} | {:6.2f} | {:6d} \n'.format(
                self.tdsData.minFreq, self.tdsData.spanFreq,
                self.tdsData.imFreq, self.tdsData.adcCLK*1e-9,
                self.tdsData.acqN)
        # fft info
        hd += 'FFT start | stop | zero-padding \n'
        _, i_min, i_max = check_fft_range(self.fftBox.fftMin(),
            self.fftBox.fftMax(), self.tdsData.acqN)
        # if i_max > acqN, record acqN in the header
        # because this is what actually happens
        hd += '{:9d} | {:4d} | {:12d} \n'.format(
                i_min, min(i_max, self.tdsData.acqN), self.fftBox.zeroPadding())
        # filter info
        w = self.filterBox.filterChoose.currentText()
        hd += 'Window type: {:s}\n'.format(w)
        if w == 'Maxwell-Boltzmann':
            hd += 'w(x) ~ x^A * exp(-B * x * 1e-3) \n'
            A = float(self.filterBox.aInput.text())
            B = float(self.filterBox.bInput.text())
            hd += 'A={:f} | B={:f} \n'.format(A, B)
            hd += '{:s}'.format('-'*50)
        else:
            hd += '{:s} \n{:s} \n{:s}'.format('-'*50, '-'*50, '-'*50)

        return hd

    def on_exit(self):
        self.close()

    def closeEvent(self, event):
        q = QtWidgets.QMessageBox.question(self, 'Quit？',
                       'Are you sure to quit？', QtWidgets.QMessageBox.Yes |
                       QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.Yes)
        if q == QtWidgets.QMessageBox.Yes:
            self.close()
        else:
            event.ignore()


class InfoBox(QtWidgets.QGroupBox):
    '''
        Scan information box
    '''

    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.parent = parent

        self.setTitle('Scan Information')
        self.setAlignment(QtCore.Qt.AlignLeft)

        self.filenameLabel = QtWidgets.QLabel()
        self.filenameLabel.setStyleSheet('color: #0b4495; font: bold; font-size: 14px')
        self.minFreqLabel = QtWidgets.QLabel()
        self.maxFreqLabel = QtWidgets.QLabel()
        self.detFreqLabel = QtWidgets.QLabel()
        self.spanFreqLabel = QtWidgets.QLabel()
        self.imFreqLabel = QtWidgets.QLabel()
        self.adcCLKLabel = QtWidgets.QLabel()
        self.pulseLenLabel = QtWidgets.QLabel()
        self.acqNLabel = QtWidgets.QLabel()
        self.acqTLabel = QtWidgets.QLabel()
        self.acqAvgLabel = QtWidgets.QLabel()
        self.repRateLabel = QtWidgets.QLabel()

        thisLayout = QtWidgets.QGridLayout()
        thisLayout.addWidget(self.filenameLabel, 0, 0, 1, 2)
        thisLayout.addWidget(QtWidgets.QLabel('Frequency MIN: '), 1, 0)
        thisLayout.addWidget(QtWidgets.QLabel('Frequency MAX: '), 2, 0)
        thisLayout.addWidget(QtWidgets.QLabel('Detection Freq: '), 3, 0)
        thisLayout.addWidget(QtWidgets.QLabel('Chirp Range: '), 4, 0)
        thisLayout.addWidget(QtWidgets.QLabel('Intermediate Freq: '), 5, 0)
        thisLayout.addWidget(QtWidgets.QLabel('ADC Clock: '), 6, 0)
        thisLayout.addWidget(QtWidgets.QLabel('Pulse Length: '), 7, 0)
        thisLayout.addWidget(QtWidgets.QLabel('Acq Number: '), 8, 0)
        thisLayout.addWidget(QtWidgets.QLabel('Acq Time: '), 9, 0)
        thisLayout.addWidget(QtWidgets.QLabel('Averages: '), 10, 0)
        thisLayout.addWidget(QtWidgets.QLabel('Rep Rate: '), 11, 0)
        thisLayout.addWidget(self.minFreqLabel, 1, 1)
        thisLayout.addWidget(self.maxFreqLabel, 2, 1)
        thisLayout.addWidget(self.detFreqLabel, 3, 1)
        thisLayout.addWidget(self.spanFreqLabel, 4, 1)
        thisLayout.addWidget(self.imFreqLabel, 5, 1)
        thisLayout.addWidget(self.adcCLKLabel, 6, 1)
        thisLayout.addWidget(self.pulseLenLabel, 7, 1)
        thisLayout.addWidget(self.acqNLabel, 8, 1)
        thisLayout.addWidget(self.acqTLabel, 9, 1)
        thisLayout.addWidget(self.acqAvgLabel, 10, 1)
        thisLayout.addWidget(self.repRateLabel, 11, 1)
        self.setLayout(thisLayout)


    def refresh(self):
        ''' Update scan information.
            Line frequencies are in MHz & format directly.
            Other frequencies (clock, pulse, etc.) are in SI units and use pg.siFormat() to format.
        '''

        self.filenameLabel.setText(self.parent.tdsData.tdsFileName)
        self.minFreqLabel.setText('{:.2f} MHz'.format(self.parent.tdsData.minFreq))
        self.maxFreqLabel.setText('{:.2f} MHz'.format(self.parent.tdsData.maxFreq))
        self.detFreqLabel.setText('{:.2f} MHz'.format(self.parent.tdsData.detFreq))
        self.spanFreqLabel.setText('{:.1f} MHz'.format(self.parent.tdsData.spanFreq))
        self.imFreqLabel.setText('{:.1f} MHz'.format(self.parent.tdsData.imFreq))
        self.adcCLKLabel.setText(pg.siFormat(self.parent.tdsData.adcCLK, precision=3, suffix='Hz'))
        self.pulseLenLabel.setText(pg.siFormat(self.parent.tdsData.pulseLen, precision=4, suffix='s'))
        self.acqNLabel.setText(str(self.parent.tdsData.acqN))
        self.acqTLabel.setText(pg.siFormat(self.parent.tdsData.acqT, precision=4, suffix='s'))
        self.acqAvgLabel.setText(str(self.parent.tdsData.acqAvg))
        self.repRateLabel.setText('{:g} μs'.format(self.parent.tdsData.repRate))


class FFTBox(QtWidgets.QGroupBox):
    '''
        FFT setting box
    '''

    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.parent = parent

        self.setTitle('FFT Setting')
        self.setAlignment(QtCore.Qt.AlignLeft)
        self.setDisabled(True)  # disable the box unless tds is loaded

        # fft range
        self.fftMinInput = QtWidgets.QLineEdit()
        self.fftMaxInput = QtWidgets.QLineEdit()
        self.fftMinInput.textChanged.connect(self._refresh_min)
        self.fftMaxInput.textChanged.connect(self._refresh_max)
        self.fftMinInput.editingFinished.connect(self.parent.calc)
        self.fftMaxInput.editingFinished.connect(self.parent.calc)
        self.fftMinTime = QtWidgets.QLabel()
        self.fftMaxTime = QtWidgets.QLabel()
        # zero padding coeff
        self.zeroPaddingCheck = QtWidgets.QCheckBox('Zero Padding')
        self.zeroPaddingCheck.setChecked(True)
        self.zeroPaddingCheck.stateChanged.connect(self.parent.calc)
        self.zeroPaddingCheck.stateChanged.connect(self._set_zero_padding)
        self.zeroPaddingInput = QtWidgets.QLineEdit('1')
        self.zeroPaddingInput.setValidator(QtGui.QIntValidator(1, 9))
        self.zeroPaddingInput.editingFinished.connect(self.parent.calc)
        # freq cutoff on display & data saving
        self.limitFreqCheck = QtWidgets.QCheckBox('Restrict spectrum to chirp frequency')
        self.limitFreqCheck.stateChanged.connect(self._set_cut_low)
        self.limitFreqCheck.stateChanged.connect(self.parent.fdsPlot)
        self.cutLowInput = QtWidgets.QLineEdit('20')
        self.cutLowInput.setValidator(QtGui.QDoubleValidator(0, 30, 1))
        self.cutLowInput.editingFinished.connect(self.parent.calc)
        self._set_cut_low()

        thisLayout = QtWidgets.QGridLayout()
        thisLayout.addWidget(QtWidgets.QLabel('FFT Range'), 0, 0)
        thisLayout.addWidget(QtWidgets.QLabel('Point'), 0, 1)
        thisLayout.addWidget(QtWidgets.QLabel('Time'), 0, 2)
        thisLayout.addWidget(QtWidgets.QLabel('Start'), 1, 0)
        thisLayout.addWidget(QtWidgets.QLabel('Stop'), 2, 0)
        thisLayout.addWidget(self.fftMinInput, 1, 1)
        thisLayout.addWidget(self.fftMaxInput, 2, 1)
        thisLayout.addWidget(self.fftMinTime, 1, 2)
        thisLayout.addWidget(self.fftMaxTime, 2, 2)
        thisLayout.addWidget(self.zeroPaddingCheck, 3, 0, 1, 1)
        thisLayout.addWidget(self.zeroPaddingInput, 3, 1, 1, 1)
        thisLayout.addWidget(self.limitFreqCheck, 4, 0, 1, 2)
        thisLayout.addWidget(QtWidgets.QLabel('Cut Low Freq (MHz)'), 5, 0, 1, 1)
        thisLayout.addWidget(self.cutLowInput, 5, 1, 1, 1)
        self.setLayout(thisLayout)

    def _refresh_min(self):
        ''' Refresh time convertion '''

        text_min = self.fftMin()
        t_min = text_min / self.parent.tdsData.adcCLK
        self.fftMinTime.setText(pg.siFormat(t_min, precision=4, suffix='s'))

    def _refresh_max(self):
        ''' Refresh time convertion '''

        text_max = self.fftMax()
        t_max = text_max / self.parent.tdsData.adcCLK
        self.fftMaxTime.setText(pg.siFormat(t_max, precision=4, suffix='s'))

    def _set_zero_padding(self):
        ''' Enable/disable zero padding setting '''

        if self.zeroPaddingCheck.isChecked():
            self.zeroPaddingInput.setReadOnly(False)
            self.zeroPaddingInput.setStyleSheet('background-color: None')
        else:
            self.zeroPaddingInput.setReadOnly(True)
            self.zeroPaddingInput.setStyleSheet('background-color: #E0E0E0')

    def _set_cut_low(self):
        ''' Enable/disable cut low setting '''

        if self.limitFreqCheck.isChecked():
            self.cutLowInput.setReadOnly(False)
            self.cutLowInput.setStyleSheet('background-color: None')
        else:
            self.cutLowInput.setReadOnly(True)
            self.cutLowInput.setStyleSheet('background-color: #E0E0E0')

    def setInitInput(self):
        ''' Set initial fft window input from tds data '''

        data_max = self.parent.tdsData.acqN
        t_max = self.parent.tdsData.acqT
        self.fftMinInput.setText('0')
        self.fftMaxInput.setText(str(data_max))
        self.fftMinTime.setText(pg.siFormat(0, precision=4, suffix='s'))
        self.fftMaxTime.setText(pg.siFormat(t_max, precision=4, suffix='s'))
        # No longer need to specify max for the validator because the input
        # validity is accounted by check_fft_range()
        # This will allow real-time refresh even when input is invalid
        val = QtGui.QIntValidator()
        val.setBottom(0)
        self.fftMinInput.setValidator(val)
        self.fftMaxInput.setValidator(val)

    def zeroPadding(self):
        ''' Return the zero-padding coefficient '''

        if self.zeroPaddingCheck.isChecked():
            return int(self.zeroPaddingInput.text())
        else:
            return 0

    def fftMin(self):
        ''' Return the fft min value (int) '''

        if self.fftMinInput.text():
            return int(self.fftMinInput.text())
        else:
            return 0

    def fftMax(self):
        ''' Return the fft max value (int) '''

        if self.fftMaxInput.text():
            return int(self.fftMaxInput.text())
        else:
            return 0

    def cutLowFreq(self):
        ''' Return the cut-low frequency (float) in Hz '''

        if self.cutLowInput.text():
            return float(self.cutLowInput.text()) * 1e6
        else:
            return 0


class FilterBox(QtWidgets.QGroupBox):
    '''
        Filter setting box
    '''

    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.parent = parent

        self.setTitle('Filter')
        self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.setDisabled(True)  # disable the box unless tds is loaded

        self.filterChoose = QtWidgets.QComboBox()
        self.filterChoose.addItems(['None',
                                    'Maxwell-Boltzmann',
                                    'Bartlett',
                                    'Blackman',
                                    'Hamming',
                                    'Hanning'])
        self.fLabel = QtWidgets.QLabel()    # display function
        # You have to preset default values.
        # Otherwise the 1st time to run getWinF will throw error
        self.aInput = QtWidgets.QLineEdit('1') # adjusting parameter a
        self.bInput = QtWidgets.QLineEdit('1')  # adjusting parameter b
        #self.cInput = QtWidgets.QLineEdit('1') # adjusting parameter c
        self.aNote = QtWidgets.QLabel()
        self.bNote = QtWidgets.QLabel()
        self.cNote = QtWidgets.QLabel()
        self.aInput.setValidator(QtGui.QDoubleValidator())
        self.bInput.setValidator(QtGui.QDoubleValidator())
        #self.cInput.setValidator(QtGui.QDoubleValidator())
        # once option / par changed, replot window function curve
        self.filterChoose.currentTextChanged.connect(self._setWinF)
        self.filterChoose.currentTextChanged.connect(self.parent.wfPlot)
        self.filterChoose.currentTextChanged.connect(self.parent.calc)
        self.aInput.editingFinished.connect(self.parent.wfPlot)
        self.bInput.editingFinished.connect(self.parent.wfPlot)
        #self.cInput.editingFinished.connect(self.parent.wfPlot)
        self.aInput.editingFinished.connect(self.parent.calc)
        self.bInput.editingFinished.connect(self.parent.calc)
        #self.cInput.editingFinished.connect(self.parent.calc)
        self._setInputEnable(False)

        thisLayout = QtWidgets.QGridLayout()
        thisLayout.addWidget(QtWidgets.QLabel('Filter Type'), 0, 0)
        thisLayout.addWidget(self.filterChoose, 0, 1)
        thisLayout.addWidget(self.fLabel, 1, 0, 1, 3)
        thisLayout.addWidget(QtWidgets.QLabel('A = '), 2, 0)
        thisLayout.addWidget(self.aInput, 2, 1)
        thisLayout.addWidget(self.aNote, 2, 2)
        thisLayout.addWidget(QtWidgets.QLabel('B = '), 3, 0)
        thisLayout.addWidget(self.bInput, 3, 1)
        thisLayout.addWidget(self.bNote, 3, 2)
        #thisLayout.addWidget(QtWidgets.QLabel('C = '), 4, 0)
        #thisLayout.addWidget(self.cInput, 4, 1)
        #thisLayout.addWidget(self.cNote, 4, 2)
        self.setLayout(thisLayout)

    def _setWinF(self):
        ''' Set window function widgets '''

        w = self.filterChoose.currentText()

        # If Maxwell-Boltzmann, enable parameter editing, else disable
        if w == 'Maxwell-Boltzmann':
            self._setInputEnable(True)
            self.fLabel.setText('f(x) ~ x^A * exp(-B * x * 1e-3) ')
            self.aNote.setText('Position')
            self.bNote.setText('Width & Tail')
        else:
            self.fLabel.setText('')
            self._setInputEnable(False)


    def _setInputEnable(self, bool):
        ''' Set status for parameter input boxes '''

        if bool:
            self.aInput.setReadOnly(False)
            self.bInput.setReadOnly(False)
            #self.cInput.setReadOnly(False)
            self.aInput.setStyleSheet('background-color: None; color: black')
            self.bInput.setStyleSheet('background-color: None; color: black')
            #self.cInput.setStyleSheet('background-color: None; color: black')
        else:
            self.aInput.setReadOnly(True)
            self.bInput.setReadOnly(True)
            #self.cInput.setReadOnly(True)
            # Visually hide the par texts
            self.aInput.setStyleSheet('background-color: #E0E0E0; color: #E0E0E0')
            self.bInput.setStyleSheet('background-color: #E0E0E0; color: #E0E0E0')
            #self.cInput.setStyleSheet('background-color: #E0E0E0; color: #E0E0E0')
            # clear parameter note
            self.aNote.setText('')
            self.bNote.setText('')
            #self.cNote.setText('')

    def getWinF(self, n):
        ''' Return window function array '''

        w = self.filterChoose.currentText()

        if w == 'Maxwell-Boltzmann':
            x = np.arange(n)
            A = float(self.aInput.text())
            B = float(self.bInput.text())
            #C = float(self.cInput.text())
            # check par validity
            if A < 0:
                self.aInput.setStyleSheet('color: #D63333')
                return np.zeros(n)
            else:
                self.aInput.setStyleSheet('color: black')
            wf = np.power(x, A) * np.exp(-B * x * 1e-3)
            # rescale to max(wf)=1
            return wf / np.max(wf)
        else:
            if w == 'None':
                return np.ones(n)
            elif w == 'Bartlett':
                return np.bartlett(n)
            elif w == 'Blackman':
                return np.blackman(n)
            elif w == 'Hamming':
                return np.hamming(n)
            elif w == 'Hanning':
                return np.hanning(n)
            else:
                return np.ones(n)

class TDSData():
    '''
        Time domain data
    '''

    def __init__(self):
        '''
            Initiate the class
            Class attributes:
                self.isData: bool     data loading status (only needed for the 1st set of data to load)
                self.tdsFileName: str data file name
                self.tdsFileDir: str  data file directory
                self.minFreq: float   start frequency f_min (MHz)
                self.imFreq: float    intermediate frequency f_im (MHz)
                self.spanFreq: float  frequency span f_max-f_min (MHz)
                self.maxFreq: float   end frequency f_max = f_min + f_span (MHz)
                self.detFreq: float   detection frequency f_det = f_max + f_im (MHz)
                self.adcCLK: float    ADC clock frequency (Hz)
                self.pulseLen: float  pulse time (* ADC clock cycles) (s)
                self.acqAvg:  int     number of averages
                self.acqN:  int       acquisition number of points
                self.acqT:  float     acquisition time (s)
                self.repRate: float   repetition rate
                self.tdsSpec: n by 2 np.array   spectrum (xy) unit(s,V)
        '''

        self.isData = False
        self.tdsFileDir = DEFAULT_DIR
        self.tdsFileName = ''
        self.minFreq = 0
        self.spanFreq = 0
        self.imFreq = 0
        self.maxFreq = 0
        self.detFreq = 0
        self.adcCLK = 0
        self.pulseLen = 0
        self.acqAvg = 0
        self.acqN = 0
        self.acqT = 0
        self.repRate = 0
        self.tdsSpec = np.zeros((2, 1))

    def load_file(self, filename):
        '''
            Reading the data file.
            header : # 183292.10|30.00|30.00|1.00E+009|832|4096|30|4096
            Returns:
                True  - load sucessfully
                False - file not found / wrong format
        '''

        if filename:
            try:
                # Get header
                with open(filename, 'r') as f:
                    header = f.readline()
                # Load spectrum
                y = np.loadtxt(filename, skiprows=1)
                # If both commands succeeded, process everyting
                hd_array = header.split('|')
                # Write header info to class attributes
                self.minFreq = float(hd_array[0])
                self.spanFreq = float(hd_array[1])
                self.imFreq = float(hd_array[2])
                self.maxFreq = self.minFreq + self.spanFreq
                self.detFreq = self.maxFreq + self.imFreq
                self.adcCLK = float(hd_array[3])
                self.pulseLen = int(hd_array[4]) / self.adcCLK
                self.acqAvg = int(hd_array[5])
                self.acqT = float(hd_array[7]) * 1e-9
                self.acqN = round(self.acqT * self.adcCLK)
                self.repRate = float(hd_array[6])
                # The last data point is 0 but leave it (for fft purpose)
                x = np.arange(self.acqN + 1) / self.adcCLK
                self.tdsSpec = np.column_stack((x, y))
                # store file name and directory
                f_str = filename.split('/')
                self.tdsFileDir = '/'.join(f_str[-1])
                self.tdsFileName = f_str[-1]
                self.isData = True
                return True
            except:
                return False
        else:
            return False


class ABSData():
    '''
        Absorption spectrum data
    '''

    def __init__(self):
        '''
            Initiate the class.
            Class attributes:
                self.isData: bool     data loading status
                self.minFreq: float   start frequency f_min (MHz)
                self.maxFreq: float   end frequency f_max (MHz)
        '''

        self.isData = False
        self.minFreq = 0
        self.maxFreq = 0
        self.fft_f_min = 0
        self.fft_f_max = 0
        self.absSpec = np.zeros((2, 1))

    def load_file(self, filename, fft_f_min, fft_f_max):
        '''
            Reading the data file. Skip the header.
            Arguments
                fft_f_min: FFT freq min (MHz)
                fft_f_max: FFT freq max (MHz)
            Returns:
                True  - load sucessfully
                False - file not found / wrong format
        '''

        if filename:
            try:
                # Load spectrum
                spec = np.loadtxt(filename, skiprows=1)
                spec = self._adjust_range(spec, fft_f_min, fft_f_max)
                # rescale freq to si unit
                spec[:, 0] = spec[:, 0] * 1e6
                self.absSpec = spec
                self.isData = True
                return True
            except:
                self.isData = False
                return False
        else:
            self.isData = False
            return False

    def _adjust_range(self, spec, fft_f_min, fft_f_max):
        ''' Adjust spectrum range to save storage.
            The largest range is the full FFT range +/- 200 MHz,
            defined by ABS_RANGE
        '''

        # get original spectral range
        self.minFreq = np.min(spec[:, 0])
        self.maxFreq = np.max(spec[:, 0])
        # adjust range
        fmin = max(self.minFreq, fft_f_min - ABS_RANGE)
        fmax = min(self.maxFreq, fft_f_max + ABS_RANGE)
        # get index
        idx = np.logical_and(spec[:, 0]>=fmin, spec[:, 0]<=fmax)
        return spec[idx, :]


def check_fft_range(i_min, i_max, n):
    ''' check the validity of i_min and i_max for FFT range
    Arguments:
        i_min: input fft_min (int)
        i_max: input fft_max (int)
        n: max data points   (int)
    Returns:
        bool: valid / invalid
        i_min: adjusted i_min
        i_max: adjusted i_max
    '''

    if i_min >= 0 and i_min < i_max:
        # both values are valid.
        # if i_max > n, numpy will automatically truncate to the end
        return True, i_min, i_max
    else:   # invalid, set to full range
        return False, 0, n


class BatchProc(QtWidgets.QDialog):
    ''' Batch process element with a progress bar dialog window '''

    def __init__(self, parent, filenames, type):
        ''' 'type' tells which batch function to envoke '''

        QtGui.QWidget.__init__(self)
        self.parent = parent
        self.filenames = filenames
        self.filenames.reverse()
        self.type = type

        # set up dialog bar
        self.setWindowTitle('Batch Process')
        self.progBar = QtWidgets.QProgressBar()
        self.progBar.setValue(0)
        self.progBar.setRange(0, len(filenames))
        thisLayout = QtGui.QVBoxLayout()
        thisLayout.addWidget(self.progBar)
        self.setLayout(thisLayout)

        # set timer
        self.progTimer = QtCore.QTimer()
        self.progTimer.setInterval(INTVL)
        self.progTimer.setSingleShot(True)
        self.progTimer.timeout.connect(self._timer_ctrl)
        self.progTimer.start()

    def _timer_ctrl(self):
        ''' Control the timer '''

        if self.type == 'fft':
            self._proc_fft()
        elif self.type == 'fit':
            self._proc_fit()
        else:
            pass

    def _proc_fft(self):
        ''' Process batch fft with current settings '''

        if self.filenames:
            # load data file
            filename = self.filenames.pop()
            status = self.parent.tdsData.load_file(filename)
            if status:  # sucessfully load file
                # refresh info box
                self.parent.infoBox.refresh()
                # plot data on tds canvas
                self.parent.tdsCurve.setData(self.parent.tdsData.tdsSpec)
                self.parent.wfPlot()
                self.parent.calc()
                # save data with filename extension replaced to .txt
                if self.parent.fftBox.limitFreqCheck.isChecked():
                    spec = self.parent.fdsSpecLimit
                else:
                    spec = self.parent.fdsSpecFull
                # rescale the frequency to MHz unit & adjust to line frequency
                spec[:, 0] = self.parent.tdsData.detFreq - spec[:, 0] * 1e-6
                # prepare header
                hd = self.parent.getHeader()
                np.savetxt(filename.replace('.tdf', '.txt'), spec,
                           delimiter='\t', fmt='%.4f', header=hd)
            else:
                pass
            self.progBar.setValue(self.progBar.value() + 1)
            self.progTimer.start()
        else:
            pass


    def _proc_fit(self):
        ''' Process batch fit with current settings '''

        if self.filenames:
            # load data file
            filename = self.filenames.pop()
            status = self.parent.loadSingle(filename)
            if status:
                self.parent.fitParBox.autoPeak()
                self.parent.fit(depress=True)
                # prepare header
                hd = 'This spectrum is fitted by fft.py (Luyao Zou).\n'
                hd += 'Source file: ' + filename + '\n'
                hd += 'Log file: ' + filename.replace('.fit', '.log') + '\n'
                hd += 'x (MHz)  y_data  y_fit'
                data = np.column_stack((self.parent.dataX, self.parent.dataY, self.parent.fitResY))
                np.savetxt(filename[:-4]+'.fit', header=hd, fmt=['%.4f', '%8e', '%8e'])
                with open(filename[:-4]+'.log', 'w') as f:
                    f.write(lmfit.fit_report(self.fitRes))
            else:
                pass
            self.progBar.setValue(self.progBar.value() + 1)
            self.progTimer.start()
        else:
            pass


class FitDiag(QtGui.QDialog):
    ''' Dialog window for spectral fit '''

    def __init__(self, parent):

        QtGui.QWidget.__init__(self)
        self.parent = parent
        self.dataX = np.zeros(1)
        self.dataY = np.zeros(1)
        self.fitResY = np.zeros(1)
        self.dataName = ''           # data file name
        self.dataDir = DEFAULT_DIR   # data file directory
        self.fitReport = ''

        self.setWindowTitle('Fit Spectrum')
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)
        self.resize(QtCore.QSize(1200, 600))

        # Initiate component widgets\
        self._init_canvas()
        self._init_parbox()

        # Set window layout
        thisLayout = QtGui.QGridLayout()
        thisLayout.addWidget(self.fitGW, 0, 0, 3, 3)
        thisLayout.addWidget(self.parWidget, 0, 3, 3, 1)
        self.setLayout(thisLayout)

    def _init_canvas(self):
        ''' Initiate fit plot canvas '''

        self.fitGW = pg.GraphicsWindow(title='Frequency domain spectrum')
        p1 = self.fitGW.addPlot(row=0, col=0)
        p2 = self.fitGW.addPlot(row=1, col=0)
        p1.setTitle('Spectrum')
        p1.setLabel('left', text='Intensity', units='')
        p1.setLabel('bottom', text='Frequency', units='Hz')
        p1.setLabel('right')
        p1.showGrid(x=True, y=True, alpha=0.8)
        p2.setTitle('Residual')
        p2.setLabel('left', text='Intensity', units='')
        p2.setLabel('bottom', text='Frequency', units='Hz')
        p2.setLabel('right')
        p2.showGrid(x=True, y=True, alpha=0.8)
        p2.setXLink(p1)
        self.specCurve = p1.plot()
        self.specCurve.setPen(color='FFB62F', width=1.5)
        self.fitCurve = pg.PlotCurveItem()
        p1.addItem(self.fitCurve)
        self.fitCurve.setPen(color='a8d5d5', width=1.5)
        self.residCurve = p2.plot()
        self.residCurve.setPen(color='FFB62F', width=1.5)

    def _init_parbox(self):
        ''' Initiate fit parameter box widgets '''

        self.filenameLabel = QtWidgets.QLabel()
        self.filenameLabel.setStyleSheet('color: #0b4495; font: bold; font-size: 14px')
        self.filenameLabel.setMaximumHeight(30)
        self.fitParBox = FitParBox(self)
        self.openBtn = QtWidgets.QPushButton('Open')
        self.openBtn.clicked.connect(self._open)
        self.fitBtn = QtWidgets.QPushButton('Fit')
        self.fitBtn.clicked.connect(self.fit)
        self.saveBtn = QtWidgets.QPushButton('Save')
        self.saveBtn.clicked.connect(self._save_fit)
        self.batchBtn = QtWidgets.QPushButton('Batch')
        self.batchBtn.clicked.connect(self._batch_fit)

        self.parWidget = QtWidgets.QWidget()
        parLayout = QtWidgets.QGridLayout()
        parLayout.addWidget(self.openBtn, 0, 0, 1, 1)
        parLayout.addWidget(self.batchBtn, 0, 1, 1, 1)
        parLayout.addWidget(self.filenameLabel, 1, 0, 1, 2)
        parLayout.addWidget(self.fitParBox, 2, 0, 4, 2)
        parLayout.addWidget(self.fitBtn, 6, 0, 1, 1)
        parLayout.addWidget(self.saveBtn, 6, 1, 1, 1)
        self.parWidget.setLayout(parLayout)


    def fit(self, depress=False):
        ''' Fit spectrum. Depress warning dialog default = False'''

        params = self._getParams()
        try:
            minner = lmfit.Minimizer(f2min, params, fcn_args=(self.dataX, self.dataY))
            self.fitRes = minner.minimize()
            # plot residual
            self.residCurve.setData(self.dataX*1e6, self.fitRes.residual)
            # plot fit
            self.fitResY = self.dataY + self.fitRes.residual
            self.fitCurve.setData(self.dataX*1e6, self.fitResY)
            # change to fitted color: salmon
            self.fitCurve.setPen(color='e64a57', width=1.5)
        except ValueError:
            if depress:
                pass
            else:
                d = QtWidgets.QMessageBox(QtGui.QMessageBox.Warning, 'Fit fails', 'Fit fails. Please try new initial guesses.')
                d.exec_()

    def plotInit(self):
        ''' plot curve using initial guesses '''

        params = self._getParams()
        # initY = model - 0
        initY = f2min(params, self.dataX, np.zeros_like(self.dataX))
        self.fitCurve.setData(self.dataX*1e6, initY)
        # change to initial guess color: lake blue
        self.fitCurve.setPen(color='a8d5d5', width=1.5)

    def _getParams(self):
        ''' Get initial guesses of parameters from user input.
        Returns: lmfit Parameters() object.
        '''

        params = lmfit.Parameters()
        for key, value in self.fitParBox.getAllValues().items():
            params.add(key, value)
        return params

    def _open(self):
        ''' Open single frequency domain spectrum & plot it '''

        # open file dialog
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                'Open Data File', self.dataDir, 'Frequency domain spectrum (*.txt *.csv *.dat)')
        if filename:
            status = self.loadSingle(filename)
            if status:
                pass
            else:
                d = QtWidgets.QMessageBox(QtGui.QMessageBox.Warning, 'Wrong data format', 'Please select correct data files.')
                d.exec_()
        else:
            pass

    def loadSingle(self, filename):
        ''' Load single data file. Returns status '''

        try:
            # clear plot
            self.specCurve.clear()
            self.fitCurve.clear()
            self.residCurve.clear()
            data = np.loadtxt(filename, skiprows=9)
            self.dataX = data[:, 0]
            self.dataY = data[:, 1]
            # update label & plot
            _l = filename.split('/')
            self.dataDir = '/'.join(_l[:-1])
            self.dataName = _l[-1]
            self.filenameLabel.setText(self.dataName)
            # make freq unit Hz to cope with pyqtgraph
            self.specCurve.setData(self.dataX*1e6, self.dataY)
            # enable par box
            self.fitParBox.setDisabled(False)
            # reset peak
            self.fitParBox.resetPeak()
            return True
        except:
            return False

    def _save_fit(self):
        ''' Save fit spetrum & fit result. '''

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self,
                'Save Spectrum & Fit', self.dataDir, 'Spectral fit (*.fit)')

        if filename:
            # prepare header
            hd = 'This spectrum is fitted by fft.py (Luyao Zou).\n'
            hd += 'Source file: ' + filename + '\n'
            hd += 'Log file: ' + filename.replace('.fit', '.log') + '\n'
            hd += 'x (MHz)  y_data  y_fit'
            data = np.column_stack((self.dataX, self.dataY, self.fitResY))
            np.savetxt(filename, data, header=hd, fmt=['%.4f', '%8e', '%8e'])
            with open(filename.replace('.fit', '.log'), 'w') as f:
                f.write(lmfit.fit_report(self.fitRes))
        else:
            pass

    def _batch_fit(self):
        ''' Batch fit '''

        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(self,
            'Open Files to Fit', self.dataDir, 'Frequency domain spectrum (*.txt, *.csv, *.dat)')

        if filenames:
            progD = BatchProc(self, filenames, type='fit')
            progD.exec_()
        else:
            pass


class FitParBox(QtWidgets.QGroupBox):
    ''' Fit parameter box '''

    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.parent = parent

        self.setTitle('Fit parameters')
        self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        # Buttons
        self.autoPeakBtn = QtWidgets.QPushButton('Auto Peak')
        self.autoPeakBtn.clicked.connect(self.autoPeak)
        self.addPeakBtn = QtWidgets.QPushButton('Add Peak')
        self.addPeakBtn.clicked.connect(self._add_peak)
        # add all delete buttons & edit buttons to button group for tracking
        self.delBtnGroup = QtWidgets.QButtonGroup()
        self.delBtnGroup.buttonClicked[int].connect(self._del_peak)
        self.editBtnGroup = QtWidgets.QButtonGroup()
        self.editBtnGroup.buttonClicked[int].connect(self._edit_peak)
        self.currentPeakId = 0  # track the current editing peak id

        # Paramenter sliders
        self._init_par_slider()
        self.parObjList = []   # FitParSet object list

        # display area for peaks and parameter values
        self.parDispLayout = QtWidgets.QGridLayout()
        self.parDispLayout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignJustify)
        self.parDispLayout.addWidget(QtWidgets.QLabel('Fit pars'), 0, 0, 1, 2)
        self.parDispLayout.addWidget(QtWidgets.QLabel('Peak x0 (MHz)'), 0, 2)
        self.parDispLayout.addWidget(QtWidgets.QLabel('A'), 0, 3)
        self.parDispLayout.addWidget(QtWidgets.QLabel('σ'), 0, 4)
        self.parDispLayout.addWidget(QtWidgets.QLabel('γ'), 0, 5)

        parDispWidget = QtWidgets.QWidget()
        parDispWidget.setLayout(self.parDispLayout)

        parDispArea = QtWidgets.QScrollArea()
        parDispArea.setWidgetResizable(True)
        parDispArea.setWidget(parDispWidget)
        parDispArea.setMinimumWidth(370)

        # Set up main layout
        mainLayout = QtWidgets.QGridLayout()
        mainLayout.setSpacing(0)
        mainLayout.addWidget(self.addPeakBtn, 0, 0, 1, 2)
        mainLayout.addWidget(self.autoPeakBtn, 0, 2, 1, 2)
        mainLayout.addWidget(self.parSlider, 1, 0, 1, 4)
        mainLayout.addWidget(parDispArea, 2, 0, 1, 4)
        self.setLayout(mainLayout)
        self.setDisabled(True)

    def _init_par_slider(self):
        ''' Setup slider properties. The conversion from int slider position
        to actual values see _update_#_val functions
        '''

        # Fix par checkboxes. Make this also the label of the variables.
        self.x0FixCheck = QtWidgets.QCheckBox('Fix | Peak')
        self.aFixCheck = QtWidgets.QCheckBox('Fix | A')
        self.sigmaFixCheck = QtWidgets.QCheckBox('Fix | σ')
        self.gammaFixCheck = QtWidgets.QCheckBox('Fix | γ')
        self.x0FixCheck.setMaximumWidth(85)
        self.aFixCheck.setMaximumWidth(85)
        self.sigmaFixCheck.setMaximumWidth(85)
        self.gammaFixCheck.setMaximumWidth(85)

        self.x0Slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.x0Slider.setMinimumHeight(18)
        self.x0Slider.setTracking(True)
        self.x0Slider.setRange(-100, 100)
        self.x0Slider.setSingleStep(1)
        self.x0Slider.setPageStep(1)
        self.x0Slider.setSliderPosition(0)
        self.x0Slider.valueChanged[int].connect(self._update_x0Input)

        self.aMajorSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.aMajorSlider.setMinimumHeight(18)
        self.aMajorSlider.setTracking(True)
        self.aMajorSlider.setRange(-10, 9)   # 1e-10.0 -- 1e9.9
        self.aMajorSlider.setSingleStep(1)
        self.aMajorSlider.setPageStep(1)
        self.aMajorSlider.setSliderPosition(0)
        self.aMajorSlider.valueChanged.connect(self._update_aInput)

        self.aMinorSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.aMinorSlider.setMinimumHeight(18)
        self.aMinorSlider.setTracking(True)
        self.aMinorSlider.setRange(0, 9)   # digit 0-9
        self.aMinorSlider.setSingleStep(1)
        self.aMinorSlider.setPageStep(1)
        self.aMinorSlider.setSliderPosition(0)
        self.aMinorSlider.valueChanged.connect(self._update_aInput)

        self.sigmaSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sigmaSlider.setMinimumHeight(18)
        self.sigmaSlider.setTracking(True)
        self.sigmaSlider.setRange(0, 50)   # total range 0-5 MHz
        self.sigmaSlider.setSingleStep(1)
        self.sigmaSlider.setPageStep(1)
        self.sigmaSlider.setSliderPosition(10)
        self.sigmaSlider.valueChanged[int].connect(self._update_sigmaInput)

        self.gammaSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gammaSlider.setMinimumHeight(18)
        self.gammaSlider.setTracking(True)
        self.gammaSlider.setRange(0, 50)   # total range 0-5 MHz
        self.gammaSlider.setSingleStep(1)
        self.gammaSlider.setPageStep(1)
        self.gammaSlider.setSliderPosition(50)
        self.gammaSlider.valueChanged[int].connect(self._update_gammaInput)

        self.parSlider = QtWidgets.QWidget()
        thisLayout = QtWidgets.QGridLayout()
        thisLayout.addWidget(self.x0FixCheck, 0, 0, 1, 1)
        thisLayout.addWidget(self.x0Slider, 0, 1, 1, 3)
        thisLayout.addWidget(self.aFixCheck, 1, 0, 1, 1)
        thisLayout.addWidget(self.aMajorSlider, 1, 1, 1, 1)
        thisLayout.addWidget(self.aMinorSlider, 1, 2, 1, 2)
        thisLayout.addWidget(self.sigmaFixCheck, 2, 0, 1, 1)
        thisLayout.addWidget(self.sigmaSlider, 2, 1, 1, 3)
        thisLayout.addWidget(self.gammaFixCheck, 3, 0, 1, 1)
        thisLayout.addWidget(self.gammaSlider, 3, 1, 1, 3)
        self.parSlider.setLayout(thisLayout)
        self.parSlider.setDisabled(True)    # disable par sliders @ 0 peaks

    def resetPeak(self):
        ''' Reset to default one peak '''

        if self.parObjList:
            n = len(self.parObjList)
            if n>1:
                for i in range(n-1):
                    self._del_peak(i)
            else:
                pass
        else:
            self._add_peak()

    def autoPeak(self):
        ''' Auto peak algorithm '''

        # estimate width
        w = round(0.9 / np.min(np.diff(self.parent.dataX)))
        peak_pos = peak_search(self.parent.dataX, self.parent.dataY, w)
        # check number of peaks with existing peak input widgets
        n_peak = np.shape(peak_pos)[0]
        n_obj = len(self.parObjList)
        if n_peak < n_obj:   # remove extra rows
            for i in range(n_obj - n_peak):
                self._del_peak(n_obj - 1 - i)
        elif n_peak > n_obj:  # add more rows
            for i in range(n_peak - n_obj):
                self._add_peak()
        else:
            pass
        for i in range(n_peak):
            self.parObjList[i].x0Input.setText('{:.2f}'.format(peak_pos[i, 0]))
            self.parObjList[i].aInput.setText('{:g}'.format(peak_pos[i, 1]))
            # estimate line width
            self.parObjList[i].sigmaInput.setText('0.1')
            self.parObjList[i].gammaInput.setText('0.5')
        self.parent.plotInit()

    def getAllValues(self):
        ''' Get parameter values from all parameter sets.
        Returns:
            p_dict: dictionary {'parname#': parvalue}
        '''

        p_dict = {}
        for i in range(len(self.parObjList)):
            obj = self.parObjList[i]
            p_dict['x0'+str(i)] = obj.getValue('x0')
            p_dict['a'+str(i)] = obj.getValue('a')
            p_dict['sigma'+str(i)] = obj.getValue('sigma')
            p_dict['gamma'+str(i)] = obj.getValue('gamma')
        return p_dict

    def getCurrentValues(self, obj_id):
        ''' Get parameter values from the current parameter set obj.
        Returns:
            x0, a, sigma, gamma
        '''
        x0 = self.parObjList[obj_id].getValue('x0')
        a = self.parObjList[obj_id].getValue('a')
        sigma = self.parObjList[obj_id].getValue('sigma')
        gamma = self.parObjList[obj_id].getValue('gamma')
        return x0, a, sigma, gamma

    def _add_peak(self):
        ''' Manually add peak '''

        n = len(self.parObjList)    # get current peak number
        parobj = FitParSet(self)
        self.parObjList.append(parobj)
        self.parDispLayout.addWidget(parobj.delBtn, n+1, 0)
        self.parDispLayout.addWidget(parobj.editBtn, n+1, 1)
        self.parDispLayout.addWidget(parobj.x0Input, n+1, 2)
        self.parDispLayout.addWidget(parobj.aInput, n+1, 3)
        self.parDispLayout.addWidget(parobj.sigmaInput, n+1, 4)
        self.parDispLayout.addWidget(parobj.gammaInput, n+1, 5)
        self.delBtnGroup.addButton(parobj.delBtn, n)
        self.editBtnGroup.addButton(parobj.editBtn, n)
        # the order of the following codes shall not be swtiched because
        # that will cause wrong currentPeakId tracking
        # focus to the current peak
        self._change_focus(n)
        self._update_x0Input(0) # center of the spectral band
        self._reset_slider(self.getCurrentValues(n))
        # enable par slider if this is the first peak added (n=0)
        if not n:
            self.parSlider.setDisabled(False)
        else:
            pass

    def _del_peak(self, btn_id):
        ''' Delete one peak from the peak list '''

        # If the button to be deleted is currently on focus,
        # and it is not the last button, move button focus to the next one;
        # if it is the last button, move button focus to the prev one
        if btn_id == self.currentPeakId:
            if btn_id == len(self.parObjList)-1:    # last button
                self._edit_peak(btn_id - 1)
            else:
                self._edit_peak(btn_id + 1)
        else:
            pass
        # get the peak par object
        obj = self.parObjList[btn_id]
        # remove buttons from button group
        self.delBtnGroup.removeButton(obj.delBtn)
        self.editBtnGroup.removeButton(obj.editBtn)
        # delete it from the list & it's children widgets
        del self.parObjList[btn_id]
        obj.delSet()
        obj.deleteLater()
        # check if all buttons are deleted
        if self.parObjList:
            # reset button ids > btn_id
            # by doing so, btn_id is kept the same as list index
            for obj in self.parObjList[btn_id:]:
                self.delBtnGroup.setId(obj.delBtn, self.delBtnGroup.id(obj.delBtn) - 1)
                self.editBtnGroup.setId(obj.editBtn, self.editBtnGroup.id(obj.editBtn) - 1)
            if self.currentPeakId > btn_id:
                self.currentPeakId -= 1
        else:
            # disable par slider
            self.parSlider.setDisabled(True)
            self.currentPeakId = 0

    def _edit_peak(self, btn_id):
        ''' Get the current focused peak from the peak list, and update sliders.
            Value adjustments are only allowed by dragging sliders.
        '''
        if btn_id == self.currentPeakId:
            pass
        else:
            # change focus
            self._change_focus(btn_id)
            # get current par values & reset slider position
            self._reset_slider(self.getCurrentValues(btn_id))

    def _update_x0Input(self, val):
        ''' Update x0 input value from slider position
            x0 = pos / 200 * range(xdata) + mean(xdata)
        '''
        # calculate x0
        x0 = val / 200 * np.ptp(self.parent.dataX) + np.median(self.parent.dataX)
        self.parObjList[self.currentPeakId].x0Input.setText('{:.1f}'.format(x0))
        # replot initial guess
        self.parent.plotInit()

    def _update_aInput(self):
        ''' Update a input value from slider position
            a = 10^(major_pos + minor_pos*0.1)
        '''
        major_pos = self.aMajorSlider.sliderPosition()
        minor_pos = self.aMinorSlider.sliderPosition()
        a = np.power(10, major_pos + minor_pos*0.1)
        self.parObjList[self.currentPeakId].aInput.setText('{:.1e}'.format(a))
        # replot initial guess
        self.parent.plotInit()

    def _update_sigmaInput(self, val):
        ''' Update sigma input value from slider position.
            This applies to all par objects
            sigma = pos * 0.1
        '''
        for obj in self.parObjList:
            obj.sigmaInput.setText('{:.1f}'.format(val*0.1))
        # replot initial guess
        self.parent.plotInit()

    def _update_gammaInput(self, val):
        ''' Update gamma input value from slider position.
            This applies to all par objects
            gamma = pos * 0.1
        '''
        for obj in self.parObjList:
            obj.gammaInput.setText('{:.1f}'.format(val*0.1))
        # replot initial guess
        self.parent.plotInit()

    def _change_focus(self, btn_id):
        ''' change the background color to display the current focused peak '''

        self.currentPeakId = btn_id
        for i in range(len(self.parObjList)):
            obj = self.parObjList[i]
            if i == btn_id:
                obj.x0Input.setStyleSheet('background-color: none')
                obj.aInput.setStyleSheet('background-color: none')
                obj.sigmaInput.setStyleSheet('background-color: none')
                obj.gammaInput.setStyleSheet('background-color: none')
            else:   # disable edit
                obj.x0Input.setStyleSheet('background-color: #E0E0E0')
                obj.aInput.setStyleSheet('background-color: #E0E0E0')
                obj.sigmaInput.setStyleSheet('background-color: #E0E0E0')
                obj.gammaInput.setStyleSheet('background-color: #E0E0E0')

    def _reset_slider(self, *args):
        ''' Reset slider position according to *args=(x0, a, sigma, gamma) '''

        # calculate x0 position
        a0, a1, a2, a3 = args[0]
        x0pos = np.round((a0 - np.median(self.parent.dataX)) / np.ptp(self.parent.dataX) * 200)
        self.x0Slider.setSliderPosition(x0pos)
        # calculate a major and a minor (using string trick)
        adb = '{:+.1f}'.format(np.log10(a1))
        self.aMajorSlider.setSliderPosition(int(adb[:2]))
        self.aMinorSlider.setSliderPosition(int(adb[3]))
        # calculate sigma
        self.sigmaSlider.setSliderPosition(round(a2*10))
        # calculate gamma
        self.gammaSlider.setSliderPosition(round(a3*10))

class FitParSet(QtWidgets.QWidget):
    ''' Fit parameter set objects '''

    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.parent = parent

        self.delBtn = QtWidgets.QPushButton('Del')
        self.editBtn = QtWidgets.QPushButton('Edit')
        self.x0Input = QtWidgets.QLineEdit()
        self.aInput = QtWidgets.QLineEdit()
        self.sigmaInput = QtWidgets.QLineEdit()
        self.gammaInput = QtWidgets.QLineEdit()

        # Value adjustments are only allowed by dragging sliders.
        # Manual input is disabled throughout the program
        self.x0Input.setReadOnly(True)
        self.aInput.setReadOnly(True)
        self.sigmaInput.setReadOnly(True)
        self.gammaInput.setReadOnly(True)

        self.delBtn.setMaximumWidth(45)
        self.editBtn.setMaximumWidth(45)
        self.x0Input.setMaximumWidth(90)
        self.aInput.setMaximumWidth(90)
        self.sigmaInput.setMaximumWidth(35)
        self.gammaInput.setMaximumWidth(35)

        self.x0Input.setAlignment(QtCore.Qt.AlignRight)
        self.aInput.setAlignment(QtCore.Qt.AlignRight)
        self.sigmaInput.setAlignment(QtCore.Qt.AlignRight)
        self.gammaInput.setAlignment(QtCore.Qt.AlignRight)

        self.x0Input.setValidator(QtGui.QDoubleValidator())
        self.aInput.setValidator(QtGui.QDoubleValidator())
        self.sigmaInput.setValidator(QtGui.QDoubleValidator())
        self.gammaInput.setValidator(QtGui.QDoubleValidator())

        # set default values (except for x0)
        self.aInput.setText('{:.1e}'.format(1))
        self.sigmaInput.setText('0.1')
        self.gammaInput.setText('0.5')

    def delSet(self):

        self.x0Input.clear()
        self.aInput.clear()
        self.sigmaInput.clear()
        self.gammaInput.clear()
        self.x0Input.deleteLater()
        self.aInput.deleteLater()
        self.sigmaInput.deleteLater()
        self.gammaInput.deleteLater()
        self.delBtn.deleteLater()
        self.editBtn.deleteLater()

    def getValue(self, parname):
        ''' Return parameter values for the given parname '''

        if parname == 'x0':
            t = self.x0Input.text()
            if t:
                return float(t)
            else:
                return 1
        elif parname == 'a':
            t = self.aInput.text()
            if t:
                return float(t)
            else:
                return 1
        elif parname == 'sigma':
            t = self.sigmaInput.text()
            if t:
                return float(t)
            else:
                return 1
        elif parname == 'gamma':
            t = self.gammaInput.text()
            if t:
                return float(t)
            else:
                return 1
        else:
            return 1


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
