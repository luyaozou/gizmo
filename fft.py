#! encoding = utf-8

''' This script is used for performing basic fft transformation
and filtering for the PhLAM mm chirped pulse spectrum '''

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
        self.setStyleSheet('font-size: 10pt; font-family: default')

        # Set global window properties
        self.setWindowTitle('Chirped Pulse FFT')
        self.setMinimumWidth(600)
        self.setMinimumHeight(600)
        self.resize(QtCore.QSize(900, 800))

        # initiate component classes
        self._init_menubar()
        self._init_canvas()
        self._init_parbox()
        self.tdsData = TDSData()

        # Set window layout
        self.mainLayout = QtWidgets.QHBoxLayout()
        self.mainLayout.setSpacing(6)
        #self.mainLayout.addWidget(self.taskEditor)
        #self.mainLayout.addWidget(self.banEditor)
        self.mainLayout.addWidget(self.canvasBox)
        self.mainLayout.addWidget(self.parBox)

        # Enable main window
        self.mainWidget = QtWidgets.QWidget()
        self.mainWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.mainWidget)

    def _init_menubar(self):
        ''' Initiate menu bar '''

        self.openFileAction = QtGui.QAction('Open File', self)
        self.openFileAction.setShortcut('Ctrl+O')
        self.openFileAction.triggered.connect(self._open_file)
        self.batchAction = QtGui.QAction('Batch Process', self)
        self.batchAction.setShortcut('Ctrl+Shift+B')
        self.batchAction.triggered.connect(self._batch)

        self.statusBar()

        menuFile = self.menuBar().addMenu('&File')
        menuFile.addAction(self.openFileAction)
        menuFile.addAction(self.batchAction)

    def _init_canvas(self):
        ''' initiate plot canvas '''

        # for time domain spectrum
        tdsCanvas = pg.PlotWidget(title='Time domain spectrum')
        tdsCanvas.setLabel('left', text='Voltage', units='V')
        tdsCanvas.setLabel('bottom', text='Time', units='s', unitPrefix='n')
        tdsCanvas.showGrid(x=True, y=True, alpha=0.5)
        self.tdsCurve = tdsCanvas.plot()

        # freq domain spectrum
        fdsCanvas = pg.PlotWidget(title='Frequency domain spectrum')
        fdsCanvas.setLabel('left', text='Intensity', units='a.u.')
        fdsCanvas.setLabel('bottom', text='Frequency', units='Hz', unitPrefix='M')
        fdsCanvas.showGrid(x=True, y=True, alpha=0.5)
        self.fdsCurve = fdsCanvas.plot()

        self.canvasBox = QtWidgets.QWidget()
        canvasLayout = QtWidgets.QVBoxLayout()  # layout for canvas
        canvasLayout.addWidget(tdsCanvas)  # time domain spectrum canvas
        canvasLayout.addWidget(fdsCanvas)  # freq domain spectrum canvas
        self.canvasBox.setLayout(canvasLayout)

    def _init_parbox(self):
        ''' initiate parameter box widgets '''

        self.parBox = QtWidgets.QWidget()
        parLayout = QtWidgets.QVBoxLayout()    # layout for parameters
        self.infoBox = InfoBox(self)    # for scan info (from file header)
        self.fftBox = FFTBox(self)      # for fft window & other settings
        self.filterBox = FilterBox(self)    # for filter settings
        self.okBtn = QtWidgets.QPushButton('ok')
        self.okBtn.clicked.connect(self._calc)
        self.batchBtn = QtWidgets.QPushButton('Batch')
        self.batchBtn.clicked.connect(self._batch)
        # disable the button unless data file is loaded
        self.okBtn.setDisabled(True)
        self.batchBtn.setDisabled(True)

        # add widgets & set up layout
        parLayout.addWidget(self.infoBox)
        parLayout.addWidget(self.fftBox)
        parLayout.addWidget(self.filterBox)
        parLayout.addWidget(self.okBtn)
        parLayout.addWidget(self.batchBtn)
        self.parBox.setLayout(parLayout)

    def _open_file(self):
        ''' Open a single data file '''

        # open file dialog
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                'Open Data File', '/home/luyao/Documents/Data', 'Time domain spectrum (*.tdf)')
        # load data file
        status = self.tdsData.load_file(filename)
        if status:  # sucessfully load file
            # refresh info box
            self.infoBox.refresh()
            # plot data on tds canvas
            self.tdsCurve.setData(self.tdsData.tdsSpec)
            # update lower & upper limits for the fft window
            self.fftBox.setInitInput()
            # enable "ok" button
            self.okBtn.setDisabled(False)
        else:
            pass

    def _calc(self):
        ''' Calculate fft '''

        # Chop y according to fft setting.
        i_min = self.fftBox.fftMin()
        i_max = self.fftBox.fftMax()
        y = self.tdsData.tdsSpec[i_min:i_max,1]
        # fft
        fft_y = np.fft.rfft(y)
        self.fdsCurve.setData(np.absolute(fft_y))


    def _batch(self):
        ''' Batch Process '''

        print('batch')

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


class InfoBox(QtGui.QGroupBox):
    '''
        Scan information box
    '''

    def __init__(self, parent):
        QtGui.QWidget.__init__(self, parent)
        self.parent = parent

        self.setTitle('Scan Information')
        self.setAlignment(QtCore.Qt.AlignLeft)

        self.minFreqLabel = QtWidgets.QLabel()
        self.maxFreqLabel = QtWidgets.QLabel()
        self.detFreqLabel = QtWidgets.QLabel()
        self.spanFreqLabel = QtWidgets.QLabel()
        self.imFreqLabel = QtWidgets.QLabel()
        self.adcCLKLabel = QtWidgets.QLabel()
        self.pulseLenLabel = QtWidgets.QLabel()
        self.acqNLabel = QtWidgets.QLabel()
        self.acqTLabel = QtWidgets.QLabel()
        self.repRateLabel = QtWidgets.QLabel()

        thisLayout = QtWidgets.QGridLayout()
        thisLayout.addWidget(QtWidgets.QLabel('Frequency MIN: '), 0, 0)
        thisLayout.addWidget(QtWidgets.QLabel('Frequency MAX: '), 1, 0)
        thisLayout.addWidget(QtWidgets.QLabel('Detection Freq: '), 2, 0)
        thisLayout.addWidget(QtWidgets.QLabel('Chirp Range: '), 3, 0)
        thisLayout.addWidget(QtWidgets.QLabel('Intermediate Freq: '), 4, 0)
        thisLayout.addWidget(QtWidgets.QLabel('ADC Clock: '), 5, 0)
        thisLayout.addWidget(QtWidgets.QLabel('Pulse Length: '), 6, 0)
        thisLayout.addWidget(QtWidgets.QLabel('Acq Number: '), 7, 0)
        thisLayout.addWidget(QtWidgets.QLabel('Acq Time: '), 8, 0)
        thisLayout.addWidget(QtWidgets.QLabel('Rep Rate: '), 9, 0)
        thisLayout.addWidget(self.minFreqLabel, 0, 1)
        thisLayout.addWidget(self.maxFreqLabel, 1, 1)
        thisLayout.addWidget(self.detFreqLabel, 2, 1)
        thisLayout.addWidget(self.spanFreqLabel, 3, 1)
        thisLayout.addWidget(self.imFreqLabel, 4, 1)
        thisLayout.addWidget(self.adcCLKLabel, 5, 1)
        thisLayout.addWidget(self.pulseLenLabel, 6, 1)
        thisLayout.addWidget(self.acqNLabel, 7, 1)
        thisLayout.addWidget(self.acqTLabel, 8, 1)
        thisLayout.addWidget(self.repRateLabel, 9, 1)
        self.setLayout(thisLayout)


    def refresh(self):
        ''' Update scan information.
            Line frequencies are in MHz & format directly.
            Other frequencies (clock, pulse, etc.) are in SI units and use pg.siFormat() to format.
        '''

        self.minFreqLabel.setText('{:.2f} MHz'.format(self.parent.tdsData.minFreq))
        self.maxFreqLabel.setText('{:.2f} MHz'.format(self.parent.tdsData.maxFreq))
        self.detFreqLabel.setText('{:.2f} MHz'.format(self.parent.tdsData.detFreq))
        self.spanFreqLabel.setText('{:.1f} MHz'.format(self.parent.tdsData.spanFreq))
        self.imFreqLabel.setText('{:.1f} MHz'.format(self.parent.tdsData.imFreq))
        self.adcCLKLabel.setText(pg.siFormat(self.parent.tdsData.adcCLK, precision=3, suffix='Hz'))
        self.pulseLenLabel.setText(pg.siFormat(self.parent.tdsData.pulseLen, precision=4, suffix='s'))
        self.acqNLabel.setText(str(self.parent.tdsData.acqN))
        self.acqTLabel.setText(pg.siFormat(self.parent.tdsData.acqT, precision=4, suffix='s'))
        self.repRateLabel.setText('{:g}'.format(self.parent.tdsData.repRate))


class FFTBox(QtGui.QGroupBox):
    '''
        FFT setting box
    '''

    def __init__(self, parent):
        QtGui.QWidget.__init__(self, parent)
        self.parent = parent

        self.setTitle('FFT Setting')
        self.setAlignment(QtCore.Qt.AlignLeft)

        self.fftMinInput = QtWidgets.QLineEdit()
        self.fftMaxInput = QtWidgets.QLineEdit()
        self.fftMinInput.textChanged.connect(self._refresh_min)
        self.fftMaxInput.textChanged.connect(self._refresh_max)
        self.fftMinTime = QtWidgets.QLabel()
        self.fftMaxTime = QtWidgets.QLabel()

        thisLayout = QtWidgets.QGridLayout()
        thisLayout.addWidget(QtWidgets.QLabel('FFT Window'), 0, 0)
        thisLayout.addWidget(QtWidgets.QLabel('Point'), 0, 1)
        thisLayout.addWidget(QtWidgets.QLabel('Time'), 0, 2)
        thisLayout.addWidget(QtWidgets.QLabel('Start'), 1, 0)
        thisLayout.addWidget(QtWidgets.QLabel('Stop'), 2, 0)
        thisLayout.addWidget(self.fftMinInput, 1, 1)
        thisLayout.addWidget(self.fftMaxInput, 2, 1)
        thisLayout.addWidget(self.fftMinTime, 1, 2)
        thisLayout.addWidget(self.fftMaxTime, 2, 2)
        self.setLayout(thisLayout)

    def _refresh_min(self):
        ''' Update lower limit for the input validator
            Refresh time convertion
        '''

        text_min = self.fftMin()
        data_max = self.parent.tdsData.acqN
        # reset fftMaxInput validator
        self.fftMaxInput.setValidator(QtGui.QIntValidator(text_min, data_max))
        # reset fftMinTime
        t_min = text_min / self.parent.tdsData.adcCLK
        self.fftMinTime.setText(pg.siFormat(t_min, precision=4, suffix='s'))

    def _refresh_max(self):
        ''' Update upper limit for the input validator
            Refresh time convertion
        '''

        text_max = self.fftMax()
        # reset fftMinInput validator
        self.fftMinInput.setValidator(QtGui.QIntValidator(0, text_max))
        # reset fftMaxTime
        t_max = text_max / self.parent.tdsData.adcCLK
        self.fftMaxTime.setText(pg.siFormat(t_max, precision=4, suffix='s'))

    def setInitInput(self):
        ''' Set initial fft window input from tds data '''

        data_max = self.parent.tdsData.acqN - 1 # for index
        t_max = self.parent.tdsData.acqT
        self.fftMinInput.setText('0')
        self.fftMaxInput.setText(str(data_max))
        self.fftMinTime.setText(pg.siFormat(0, precision=4, suffix='s'))
        self.fftMaxTime.setText(pg.siFormat(t_max, precision=4, suffix='s'))
        self.fftMinInput.setValidator(QtGui.QIntValidator(0, data_max))
        self.fftMaxInput.setValidator(QtGui.QIntValidator(0, data_max))

    def fftMin(self):
        ''' Return the fft min value (int) '''

        if self.fftMinInput.text():
            return int(self.fftMinInput.text())
        else:
            return 1

    def fftMax(self):
        ''' Return the fft max value (int) '''

        if self.fftMaxInput.text():
            return int(self.fftMaxInput.text())
        else:
            return 2


class FilterBox(QtGui.QGroupBox):
    '''
        Filter setting box
    '''

    def __init__(self, parent):
        QtGui.QWidget.__init__(self, parent)
        self.parent = parent

        self.setTitle('Filter')
        self.setAlignment(QtCore.Qt.AlignLeft)

        self.filterChoose = QtWidgets.QComboBox()
        self.filterChoose.addItems(['Test filter1', 'Test filter2'])

        thisLayout = QtWidgets.QGridLayout()
        thisLayout.addWidget(QtWidgets.QLabel('Filter Type'), 0, 0)
        thisLayout.addWidget(self.filterChoose, 0, 1)
        self.setLayout(thisLayout)


class TDSData():
    '''
        Time domain data
    '''

    def __init__(self):
        '''
            Initiate the class
            Class attributes:
                self.minFreq: float   start frequency f_min (MHz)
                self.imFreq: float    intermediate frequency f_im (MHz)
                self.spanFreq: float  frequency span f_max-f_min (MHz)
                self.maxFreq: float   end frequency f_max = f_min + f_span (MHz)
                self.detFreq: float   detection frequency f_det = f_max + f_im (MHz)
                self.adcCLK: float    ADC clock frequency (Hz)
                self.pulseLen: float  pulse time (* ADC clock cycles) (s)
                self.acqN:  int       acquisition data points
                self.acqT:  float     acquisition time (s)
                self.repRate: float   repetition rate
                self.tdsSpec: n by 2 np.array   spectrum (xy) unit(s,V)
        '''

        self.minFreq = 0
        self.spanFreq = 0
        self.imFreq = 0
        self.maxFreq = 0
        self.detFreq = 0
        self.adcCLK = 0
        self.pulseLen = 0
        self.acqN = 0
        self.acqT = 0
        self.repRate = 0
        self.tdsSpec = np.zeros((2, 1))

    def load_file(self, filename):
        '''
            Reading the data file.
            header : # 153292.10|30.00|30.00|1.00E+009|832|4096|30|4096
            Returns:
                True  - load sucessfully
                False - file not found / wrong format
        '''

        if filename:
            try:
                # Get header
                with open(filename, 'r') as f:
                    header = f.readline()
                hd_array = header.split('|')
                # Write header info to class attributes
                self.minFreq = float(hd_array[0])
                self.spanFreq = float(hd_array[1])
                self.imFreq = float(hd_array[2])
                self.maxFreq = self.minFreq + self.spanFreq
                self.detFreq = self.maxFreq + self.imFreq
                self.adcCLK = float(hd_array[3])
                self.pulseLen = int(hd_array[4]) / self.adcCLK
                self.acqN = int(hd_array[5])
                self.acqT = self.acqN / self.adcCLK
                self.repRate = float(hd_array[6])
                # Load spectrum
                y = np.loadtxt(filename, skiprows=1)
                # The last data point is 0.
                # It is a fake thing and needs to be chopped off
                x = np.arange(self.acqN) / self.adcCLK
                self.tdsSpec = np.column_stack((x, y[:-1]))
                return True
            except:
                return False
        else:
            return False



if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
