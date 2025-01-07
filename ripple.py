# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys
import os
from PyQt6 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
import platform
import ctypes
import numpy as np
from dataclasses import dataclass
import json


def mat_distance(dim, x0, y0):
    """ Create a distance matrix centered at (x0, y0) with dimensions dim x dim
    :argument
        dim: int, the dimension of the matrix
        x0: int, the x-coordinate of the center
        y0: int, the y-coordinate of the center
    :return
        dist: numpy.ndarray, the distance matrix
    """
    x, y = np.ogrid[:dim, :dim]
    dist = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    return dist


def wave(hyper_res, dim, x0, y0, amp, phase, lambda_, damp):
    """ Create a wave centered at (x0, y0) with amplitude and phase filled to map of dim x dim
    :argument
        hyper_res: int, hyper resolution factor
        dim: int, the dimension of the matrix
        x0: int, the x-coordinate of the center
        y0: int, the y-coordinate of the center
        amp: float, the amplitude of the wave
        phase: float, the phase of the wave
        lambda_: float, the wavelength of the wave
        damp: float, the damping factor
    :return
        wave: numpy.ndarray complex, the wave matrix
    """
    dist = mat_distance(dim * hyper_res, x0 * hyper_res, y0 * hyper_res) / hyper_res
    k = np.pi * 2 / lambda_
    amp_dist = np.copy(dist)
    amp_dist[x0 * hyper_res, y0 * hyper_res] = 1    # avoid division by zero
    amp_mat = amp * np.power((amp_dist - 1) * damp + 1, -2)
    wave = amp_mat * np.exp(1j * (k * dist + phase))
    return wave


def to_json(obj, filename):
    """ Serialize an object to json and save on disk
    :argument
        obj: plan object
        filename: str           filename to be saved
    """

    with open(filename, 'w') as fp:
        json.dump(_obj2dict(obj), fp, indent=2)


def from_json_(obj, filename):
    """ Load data from json. Mutable functiona and replace obj in place
    :argument
        obj: the object to write value in
        f: str          filename to load
    """
    with open(filename, 'r') as fp:
        dict_ = json.load(fp)
        _dict2obj_(obj, dict_)


def _obj2dict(obj):
    """ Convert plain object to dictionary (for json dump) """
    d = {}
    for attr in dir(obj):
        if not attr.startswith('__'):
            d[attr] = getattr(obj, attr)
    return d


def _dict2obj_(obj, dict_):
    """ Convert dictionary values back to plain obj. Mutable function
    :argument
        obj: object to be updated
        dict_: dictionary
    """

    for key, value in dict_.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, list):
                    # convert list to tuple
                    if len(v) > 0 and isinstance(v[0], list):
                        # convert list in list to tuple as well
                        value[k] = (tuple(vv) for vv in v)
                    else:
                        value[k] = tuple(v)
        setattr(obj, key, value)


@dataclass
class Prefs:

    geometry: tuple = (0, 0, 1600, 900)
    damp: float = 0.01
    dim: int = 100
    hyper_res: int = 1
    ripple_color_min: str = '#000000'
    ripple_color_max: str = '#FFFFFF'
    ripple_color_middle: str = '#808080'
    source_color_min: str = '#000000'
    source_color_max: str = '#FFFFFF'


class MainWin(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        # Set global window properties
        self.setWindowTitle('Ripple')
        self.prefs = Prefs()
        self._current_coord = None
        f = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ripple_prefs.json')
        if os.path.isfile(f):
            from_json_(self.prefs, f)
        self.ui = UI(self.prefs, self)
        self.setCentralWidget(self.ui)
        self.setGeometry(QtCore.QRect(*self.prefs.geometry))
        self.pt_source_amps = np.zeros((self.prefs.dim, self.prefs.dim))
        self.pt_source_phases = np.zeros((self.prefs.dim, self.prefs.dim))
        self.pt_source_lambdas = np.ones((self.prefs.dim, self.prefs.dim))
        self.render_pt_source()
        self.dProg = DialogProg('Rendering', parent=self)

        self.ui.sliderAmp.valueChanged.connect(self.update_amp)
        self.ui.sliderPhase.valueChanged.connect(self.update_phase)
        self.ui.sliderLambda.valueChanged.connect(self.update_lambda)
        self.ui.sliderDim.valueChanged.connect(self.change_dim)
        self.ui.btnClear.clicked[bool].connect(self.clear_pt_sources)
        self.ui.btnAmp.clicked[bool].connect(self.render_pt_source)
        self.ui.btnPhase.clicked[bool].connect(self.render_pt_source)
        self.ui.btnRender.clicked[bool].connect(self.render_ripple)
        self.ui.canvasPt.map.mouseReleased[int, int].connect(self.load_local_params)
        self.ui.canvasPt.map.mouseCtrlReleased[int, int].connect(self.add_pt_source)
        self.ui.canvasPt.map.ctrlWheelMoved[int, int, float].connect(self.wheel_amp)
        self.ui.canvasPt.map.altWheelMoved[int, int, float].connect(self.wheel_phase)
        self.ui.canvasPt.map.shiftWheelMoved[int, int, float].connect(self.wheel_lambda)
        self.ui.canvasPt.map.ctrlLeftDrag[int, int].connect(self.draw_trace_amp)
        self.ui.canvasPt.map.altLeftDrag[int, int].connect(self.draw_trace_phase)
        self.ui.canvasPt.map.shiftLeftDrag[int, int].connect(self.draw_trace_lambda)

        self._thread_render = _ThreadRender(self)
        self._thread_render.sig_complete.connect(self.dProg.accept)
        self._thread_render.sig_set_nt.connect(self.dProg.set_nt)
        self._thread_render.sig_update_prog.connect(self.dProg.update_prog)
        self._thread_render.sig_return_result.connect(self.ui.canvasRender.map.plot)
        self.dProg.btnAbort.clicked[bool].connect(self._thread_render.abort)

    def closeEvent(self, ev):
        """ Save window geometry """
        self.prefs.geometry = self.geometry().getRect()
        self.ui.fetch_params(self.prefs)
        to_json(self.prefs, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ripple_prefs.json'))
        ev.accept()

    def change_dim(self, d_):
        dim = int(d_)
        self.prefs.dim = dim
        self.clear_pt_sources()
        self.ui.canvasRender.map.plot(np.zeros((dim, dim)))

    def render_ripple(self):
        self.ui.fetch_params(self.prefs)
        self._thread_render.setup(self.prefs.hyper_res, self.prefs.dim, self.prefs.damp,
                                  self.pt_source_lambdas, self.pt_source_amps, self.pt_source_phases)
        self.dProg.show()
        self._thread_render.start()

    def render_pt_source(self):
        if self.ui.btnAmp.isChecked():
            self.ui.canvasPt.map.plot(self.pt_source_amps)
        elif self.ui.btnPhase.isChecked():
            self.ui.canvasPt.map.plot(self.pt_source_phases)
        elif self.ui.btnLambda.isChecked():
            self.ui.canvasPt.map.plot(self.pt_source_lambdas)
        else:
            pass

    def clear_pt_sources(self):
        self.pt_source_amps = np.zeros((self.prefs.dim, self.prefs.dim))
        self.pt_source_phases = np.zeros((self.prefs.dim, self.prefs.dim))
        self.pt_source_lambdas = np.ones((self.prefs.dim, self.prefs.dim))
        self.ui.canvasPt.map.plot(np.zeros((self.prefs.dim, self.prefs.dim)))

    def load_local_params(self, x0, y0):
        self._current_coord = (x0, y0)
        self.ui.lblXY.setText(f'({x0:d}, {y0:d})')
        self.ui.sliderAmp.setValue(self.pt_source_amps[x0, y0])
        self.ui.sliderPhase.setValue(self.pt_source_phases[x0, y0])
        self.ui.sliderLambda.setValue(self.pt_source_lambdas[x0, y0])

    def add_pt_source(self, x0, y0):
        self.pt_source_amps[x0, y0] = self.ui.sliderAmp.value()
        self.pt_source_phases[x0, y0] = self.ui.sliderPhase.value()
        self.pt_source_lambdas[x0, y0] = self.ui.sliderLambda.value()
        self.render_pt_source()
        self._current_coord = None
        self.ui.lblXY.setText('')

    def draw_trace_amp(self, x0, y0):
        self.ui.btnAmp.setChecked(True)
        self.pt_source_amps[x0, y0] = self.ui.sliderAmp.value()
        self.pt_source_lambdas[x0, y0] = self.ui.sliderLambda.value()
        self.pt_source_phases[x0, y0] = self.ui.sliderPhase.value()
        self.render_pt_source()

    def draw_trace_phase(self, x0, y0):
        self.ui.btnPhase.setChecked(True)
        self.pt_source_phases[x0, y0] = self.ui.sliderPhase.value()
        self.render_pt_source()

    def draw_trace_lambda(self, x0, y0):
        self.ui.btnLambda.setChecked(True)
        self.pt_source_lambdas[x0, y0] = self.ui.sliderLambda.value()
        self.render_pt_source()

    def update_amp(self, amp):
        if self._current_coord:
            x0, y0 = self._current_coord
            self.pt_source_amps[x0, y0] = amp
            self.render_pt_source()

    def update_phase(self, phase):
        if self._current_coord:
            x0, y0 = self._current_coord
            self.pt_source_phases[x0, y0] = phase
            self.render_pt_source()

    def update_lambda(self, lambda_):
        if self._current_coord:
            x0, y0 = self._current_coord
            self.pt_source_lambdas[x0, y0] = lambda_
            self.render_pt_source()

    def wheel_amp(self, x0, y0, delta_amp):
        # avoid infinite loop
        self.ui.sliderAmp.valueChanged.disconnect(self.update_amp)
        self.load_local_params(x0, y0)
        self.ui.sliderAmp.setValue(self.ui.sliderAmp.value() + delta_amp)
        self.draw_trace_amp(x0, y0)
        self.ui.sliderAmp.valueChanged.connect(self.update_amp)

    def wheel_phase(self, x0, y0, delta_phase):
        # avoid infinite loop
        self.ui.sliderPhase.valueChanged.disconnect(self.update_phase)
        self.load_local_params(x0, y0)
        self.ui.sliderPhase.setValue(self.ui.sliderPhase.value() + delta_phase)
        self.draw_trace_phase(x0, y0)
        self.ui.sliderPhase.valueChanged.connect(self.update_phase)

    def wheel_lambda(self, x0, y0, delta_lambda):
        # avoid infinite loop
        self.ui.sliderLambda.valueChanged.disconnect(self.update_lambda)
        self.load_local_params(x0, y0)
        self.ui.sliderLambda.setValue(self.ui.sliderLambda.value() + delta_lambda)
        self.draw_trace_lambda(x0, y0)
        self.ui.sliderLambda.valueChanged.connect(self.update_lambda)


class UI(QtWidgets.QWidget):

    def __init__(self, prefs, parent=None):
        super().__init__(parent)

        globalBar = QtWidgets.QGroupBox('Global Settings')
        self.sliderHyperRes = CustomSlider('Hyper Resolution', minimum=1, maximum=100, dec=0)
        self.sliderDim = CustomSlider('Dimension', minimum=10, maximum=1000, dec=0)
        self.sliderDamp = CustomSlider('Damping', minimum=0, maximum=1, dec=3)
        globalBarLayout = QtWidgets.QVBoxLayout()
        globalBarLayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        globalBarLayout.setSpacing(0)
        globalBarLayout.setContentsMargins(0, 5, 0, 5)
        globalBarLayout.addWidget(self.sliderHyperRes)
        globalBarLayout.addWidget(self.sliderDim)
        globalBarLayout.addWidget(self.sliderDamp)
        globalBar.setLayout(globalBarLayout)

        localLblLayout = QtWidgets.QHBoxLayout()
        localLblLayout.setSpacing(0)
        localLblLayout.setContentsMargins(10, 0, 10, 0)
        self.lblXY = QtWidgets.QLabel()
        self.btnLambda = QtWidgets.QPushButton('Wavelength')
        self.btnLambda.setFixedWidth(100)
        self.btnLambda.setCheckable(True)
        self.btnLambda.setChecked(False)
        self.btnAmp = QtWidgets.QPushButton('Amplitude')
        self.btnAmp.setFixedWidth(100)
        self.btnAmp.setCheckable(True)
        self.btnAmp.setChecked(True)
        self.btnPhase = QtWidgets.QPushButton('Phase')
        self.btnPhase.setFixedWidth(100)
        self.btnPhase.setCheckable(True)
        self.btnPhase.setChecked(False)
        self.btnClear = QtWidgets.QPushButton('Clear')
        self.btnClear.setFixedWidth(75)
        localLblLayout.addWidget(QtWidgets.QLabel('Coordinate: '))
        localLblLayout.addWidget(self.lblXY)
        localLblLayout.addStretch()
        localLblLayout.addWidget(self.btnLambda)
        localLblLayout.addWidget(self.btnAmp)
        localLblLayout.addWidget(self.btnPhase)
        localLblLayout.addWidget(self.btnClear)

        localBar = QtWidgets.QGroupBox('Point Source Settings')
        self.sliderLambda = CustomSlider('Wavelength', minimum=0.1, maximum=1000, dec=1)
        self.sliderAmp = CustomSlider('Amplitude', dec=3)
        self.sliderPhase = CustomSlider('Phase', minimum=0, maximum=360, dec=0)
        localBarLayout = QtWidgets.QVBoxLayout()
        localBarLayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        localBarLayout.setSpacing(0)
        localBarLayout.setContentsMargins(0, 5, 0, 5)
        localBarLayout.addLayout(localLblLayout)
        localBarLayout.addWidget(self.sliderLambda)
        localBarLayout.addWidget(self.sliderAmp)
        localBarLayout.addWidget(self.sliderPhase)
        localBar.setLayout(localBarLayout)

        self.canvasPt = CanvasPt('Point Sources', parent=self)
        self.canvasRender = CanvasRender('Rendered Ripples', parent=self)

        self.btnRender = QtWidgets.QPushButton('Render (Ctrl+R)')
        self.btnRender.setShortcut('Ctrl+R')
        self.btnRender.setFixedWidth(150)
        btnLayout = QtWidgets.QHBoxLayout()
        btnLayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        btnLayout.addWidget(self.btnRender)

        mainLayout = QtWidgets.QGridLayout()
        mainLayout.addWidget(localBar, 0, 0)
        mainLayout.addWidget(globalBar, 0, 1)
        mainLayout.addLayout(btnLayout, 1, 0, 1, 2)
        mainLayout.addWidget(self.canvasPt, 2, 0)
        mainLayout.addWidget(self.canvasRender, 2, 1)

        self.setLayout(mainLayout)

        self.load_params(prefs)

        self.btnLambda.toggled[bool].connect(self._on_btn_lambda_clicked)
        self.btnAmp.toggled[bool].connect(self._on_btn_amp_clicked)
        self.btnPhase.toggled[bool].connect(self._on_btn_phase_clicked)

    def _on_btn_amp_clicked(self, checked):
        # if btnAmp is checked, btnPhase is unchecked
        if checked:
            self.btnPhase.setChecked(False)
            self.btnLambda.setChecked(False)

    def _on_btn_phase_clicked(self, checked):
        # if btnPhase is checked, btnAmp is unchecked
        if checked:
            self.btnAmp.setChecked(False)
            self.btnLambda.setChecked(False)

    def _on_btn_lambda_clicked(self, checked):
        # if btnLambda is checked, btnAmp and btnPhase are unchecked
        if checked:
            self.btnAmp.setChecked(False)
            self.btnPhase.setChecked(False)

    def fetch_params(self, prefs):
        prefs.hyper_res = int(self.sliderHyperRes.value())
        prefs.dim = int(self.sliderDim.value())
        prefs.damp = self.sliderDamp.value()
        self.canvasPt.fetch_params(prefs)
        self.canvasRender.fetch_params(prefs)

    def load_params(self, prefs):
        self.sliderHyperRes.setValue(prefs.hyper_res)
        self.sliderDim.setValue(prefs.dim)
        self.sliderDamp.setValue(prefs.damp)
        self.canvasPt.load_params(prefs)
        self.canvasRender.load_params(prefs)


class CustomSlider(QtWidgets.QWidget):

    valueChanged = QtCore.pyqtSignal(object)

    def __init__(self, title, minimum=0., maximum=1., dec=0, orientation=QtCore.Qt.Orientation.Horizontal,
                 parent=None):
        super().__init__(parent)

        self._min = minimum
        self._max = maximum
        self._dec = dec
        self._slider = QtWidgets.QSlider(orientation)
        self._box = QtWidgets.QDoubleSpinBox()
        self._box.setMinimum(minimum)
        self._box.setMaximum(maximum)
        self._box.setDecimals(dec)
        self._box.setSingleStep(10 ** -dec)
        self._box.setFixedWidth(125)
        thisLayout = QtWidgets.QHBoxLayout()
        thisLayout.setContentsMargins(10, 0, 10, 0)
        thisLayout.addWidget(QtWidgets.QLabel(title))
        thisLayout.addWidget(self._slider)
        thisLayout.addWidget(self._box)
        self.setLayout(thisLayout)
        self._update_slider_range()
        self._slider.valueChanged[int].connect(self._update_value)
        self._box.valueChanged[float].connect(self._update_slider)
    def setMinimum(self, v):
        self._min = v
        self._box.setMinimum(v)

    def setMaximum(self, v):
        self._max = v
        self._box.setMaximum(v)

    def setDecimals(self, v):
        self._dec = v
        self._box.setDecimals(v)
        self._box.setSingleStep(10 ** -v)
        self._update_slider_range()

    def setRange(self, minimum, maximum):
        self._min = minimum
        self._max = maximum
        self._box.setMinimum(minimum)
        self._box.setMaximum(maximum)

    def setValue(self, v):
        int_value = self._to_int(v)
        self._box.setValue(v)
        self._slider.setValue(int_value)

    def value(self):
        return self._to_float(self._slider.value())

    def _to_int(self, v):
        return int((v - self._min) * 10 ** self._dec)

    def _to_float(self, value):
        return self._min + value / (10 ** self._dec)

    def _update_slider_range(self):
        self._slider.setRange(0, int(10**self._dec * (self._max - self._min)))

    def _update_value(self, int_value):
        float_value = self._to_float(int_value)
        self._box.setValue(float_value)
        self.valueChanged.emit(float_value)

    def _update_slider(self, float_value):
        """ This is invoked by the spin box value change event directly.
        Therefore, disconnect the valueChanged signal to avoid infinite loop """
        self._slider.valueChanged[int].disconnect(self._update_value)
        int_value = self._to_int(float_value)
        self._slider.setValue(int_value)
        self.valueChanged.emit(float_value)
        self._slider.valueChanged[int].connect(self._update_value)


class CustomImageView(pg.GraphicsView):
    mouseMoved = QtCore.pyqtSignal(int, int, float)
    mouseReleased = QtCore.pyqtSignal(int, int)
    mouseCtrlReleased = QtCore.pyqtSignal(int, int)
    ctrlWheelMoved = QtCore.pyqtSignal(int, int, float)
    altWheelMoved = QtCore.pyqtSignal(int, int, float)
    shiftWheelMoved = QtCore.pyqtSignal(int, int, float)
    ctrlLeftDrag = QtCore.pyqtSignal(int, int)
    altLeftDrag = QtCore.pyqtSignal(int, int)
    shiftLeftDrag = QtCore.pyqtSignal(int, int)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        viewLayout = pg.GraphicsLayout()
        self.setCentralItem(viewLayout)

        imgPlot = viewLayout.addPlot()
        self._imgItem = pg.ImageItem()
        imgPlot.addItem(self._imgItem)
        self._imgItem.getViewBox().setAspectLocked(True, ratio=1)
        imgPlot.setLabel('top')
        imgPlot.setLabel('right')
        imgPlot.getAxis('left').setStyle(showValues=True)
        imgPlot.getAxis('right').setStyle(showValues=True)
        imgPlot.getAxis('top').setStyle(showValues=True)
        imgPlot.getAxis('bottom').setStyle(showValues=True)
        imgPlot.showGrid(x=True, y=True)

        self._color_bar = pg.ColorBarItem(values=(0, 1), colorMap='turbo')
        self._color_bar.setImageItem(self._imgItem)
        viewLayout.addItem(self._color_bar, row=0, col=1)

    def _get_coord(self, p):
        """ Get coordinate of the mouse event
        :argument
            p: QPointF, the position of the mouse event
        :return
            x: int, the x-coordinate
            y: int, the y-coordinate
        """
        px = int(self._imgItem.getViewBox().mapSceneToView(p).x())
        py = int(self._imgItem.getViewBox().mapSceneToView(p).y())
        # get bounds of the image
        xmax = self._imgItem.image.shape[1]
        ymax = self._imgItem.image.shape[0]
        # this allows the valid index when mouse moves out of the image
        x = int(min(max(0, px), xmax - 1))
        y = int(min(max(0, py), ymax - 1))
        return x, y

    def mouseMoveEvent(self, ev):
        # avoid error in case of empty image
        if self._imgItem.image is None:
            ev.ignore()
        elif ev.buttons() == QtCore.Qt.MouseButton.LeftButton and ev.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
            x, y = self._get_coord(ev.position())
            self.ctrlLeftDrag.emit(x, y)
        elif ev.buttons() == QtCore.Qt.MouseButton.LeftButton and ev.modifiers() == QtCore.Qt.KeyboardModifier.AltModifier:
            x, y = self._get_coord(ev.position())
            self.altLeftDrag.emit(x, y)
        elif ev.buttons() == QtCore.Qt.MouseButton.LeftButton and ev.modifiers() == QtCore.Qt.KeyboardModifier.ShiftModifier:
            x, y = self._get_coord(ev.position())
            self.shiftLeftDrag.emit(x, y)
        else:
            x, y = self._get_coord(ev.position())
            self.mouseMoved.emit(x, y, self._imgItem.image[x, y])

    def mouseReleaseEvent(self, ev):
        if self._imgItem.image is None:
            ev.ignore()
        elif ev.button() == QtCore.Qt.MouseButton.LeftButton:
            x, y = self._get_coord(ev.position())
            if ev.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
                self.mouseCtrlReleased.emit(x, y)
            else:
                self.mouseReleased.emit(x, y)
        else:
            super().mouseReleaseEvent(ev)

    def wheelEvent(self, ev):
        if self._imgItem.image is None:
            ev.ignore()
        elif ev.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
            x, y = self._get_coord(ev.position())
            if ev.angleDelta().y() > 0:
                self.ctrlWheelMoved.emit(x, y, 0.01)
            else:
                self.ctrlWheelMoved.emit(x, y, -0.01)
        elif ev.modifiers() == QtCore.Qt.KeyboardModifier.AltModifier:
            x, y = self._get_coord(ev.position())
            if ev.angleDelta().x() > 0:
                self.altWheelMoved.emit(x, y, 10.)
            else:
                self.altWheelMoved.emit(x, y, -10.)
        elif ev.modifiers() == QtCore.Qt.KeyboardModifier.ShiftModifier:
            x, y = self._get_coord(ev.position())
            if ev.angleDelta().y() > 0:
                self.shiftWheelMoved.emit(x, y, 1)
            else:
                self.shiftWheelMoved.emit(x, y, -1)
        else:
            super().wheelEvent(ev)

    def set_color_gradiant(self, list_colors):
        n = len(list_colors) - 1    # number of colors is 1 more than the number of divisions
        colors = list((i / n, color) for i, color in enumerate(list_colors))
        color_map = pg.ColorMap(*zip(*colors))
        self._color_bar.setColorMap(color_map)

    def plot(self, img):
        self._imgItem.setImage(img)
        self._imgItem.getViewBox().setXRange(0, img.shape[0])
        self._imgItem.getViewBox().setYRange(0, img.shape[1])
        self._color_bar.setLevels((img.min(), img.max()))
        self._color_bar.setColorMap(self._color_bar.colorMap())


class CanvasPt(QtWidgets.QGroupBox):

    def __init__(self, title, parent=None):
        super().__init__(parent)

        self.setTitle(title)

        self.map = CustomImageView()
        self.map.mouseMoved.connect(self.on_mouse_moved)

        self._dColor = QtWidgets.QColorDialog()
        self._colorMin = ColorPicker(self._dColor, color='#000000')
        self._colorMax = ColorPicker(self._dColor, color='#FFFFFF')
        colorPickerLayout = QtWidgets.QHBoxLayout()
        colorPickerLayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        colorPickerLayout.addWidget(QtWidgets.QLabel('Pick Color Gradiant  |  '))
        colorPickerLayout.addWidget(QtWidgets.QLabel('Min: '))
        colorPickerLayout.addWidget(self._colorMin)
        colorPickerLayout.addItem(QtWidgets.QSpacerItem(
            10, 0, QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Minimum))
        colorPickerLayout.addWidget(QtWidgets.QLabel('Max: '))
        colorPickerLayout.addWidget(self._colorMax)

        self._info_bar = QtWidgets.QLabel()
        self._info_bar.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)

        imageLayout = QtWidgets.QHBoxLayout()
        imageLayout.addWidget(self.map)
        imageLayout.setStretch(0, 1)  # Make the ImageView take up more space
        imageLayout.setStretch(1, 0)  # Make the info bar take up less space

        thisLayout = QtWidgets.QVBoxLayout()
        thisLayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignRight)
        thisLayout.addLayout(colorPickerLayout)
        thisLayout.addLayout(imageLayout)
        thisLayout.addWidget(self._info_bar)
        self.setLayout(thisLayout)
        self.map.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

        self._colorMin.colorPicked.connect(lambda: self.map.set_color_gradiant([self._colorMin.color, self._colorMax.color]))
        self._colorMax.colorPicked.connect(lambda: self.map.set_color_gradiant([self._colorMin.color, self._colorMax.color]))

    def on_mouse_moved(self, x, y, z):
        self._info_bar.setText(f'({x:.0f}, {y:.0f}) -> {z:.2f}')

    def fetch_params(self, prefs):
        prefs.source_color_min = self._colorMin.color
        prefs.source_color_max = self._colorMax.color

    def load_params(self, prefs):
        self._colorMin.color = prefs.source_color_min
        self._colorMax.color = prefs.source_color_max
        self.map.set_color_gradiant([self._colorMin.color, self._colorMax.color])


class CanvasRender(QtWidgets.QGroupBox):

    def __init__(self, title, parent=None):
        super().__init__(parent)

        self.setTitle(title)

        self.map = CustomImageView()
        self.map.mouseMoved.connect(self.on_mouse_moved)

        self._dColor = QtWidgets.QColorDialog()
        self._colorMin = ColorPicker(self._dColor, color='#000000')
        self._colorMiddle = ColorPicker(self._dColor, color='#808080')
        self._colorMax = ColorPicker(self._dColor, color='#FFFFFF')
        colorPickerLayout = QtWidgets.QHBoxLayout()
        colorPickerLayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        colorPickerLayout.addWidget(QtWidgets.QLabel('Pick Color Gradiant  |  '))
        colorPickerLayout.addWidget(QtWidgets.QLabel('Min: '))
        colorPickerLayout.addWidget(self._colorMin)
        colorPickerLayout.addItem(QtWidgets.QSpacerItem(
            10, 0, QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Minimum))
        colorPickerLayout.addWidget(QtWidgets.QLabel('Zero: '))
        colorPickerLayout.addWidget(self._colorMiddle)
        colorPickerLayout.addItem(QtWidgets.QSpacerItem(
            10, 0, QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Minimum))
        colorPickerLayout.addWidget(QtWidgets.QLabel('Max: '))
        colorPickerLayout.addWidget(self._colorMax)

        self._info_bar = QtWidgets.QLabel()
        self._info_bar.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)

        imageLayout = QtWidgets.QHBoxLayout()
        imageLayout.addWidget(self.map)
        imageLayout.setStretch(0, 1)  # Make the ImageView take up more space
        imageLayout.setStretch(1, 0)  # Make the info bar take up less space

        thisLayout = QtWidgets.QVBoxLayout()
        thisLayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignRight)
        thisLayout.addLayout(colorPickerLayout)
        thisLayout.addLayout(imageLayout)
        thisLayout.addWidget(self._info_bar)
        self.setLayout(thisLayout)
        self.map.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

        self._colorMin.colorPicked.connect(lambda: self.map.set_color_gradiant(
            [self._colorMin.color, self._colorMiddle.color, self._colorMax.color]))
        self._colorMiddle.colorPicked.connect(lambda: self.map.set_color_gradiant(
            [self._colorMin.color, self._colorMiddle.color, self._colorMax.color]))
        self._colorMax.colorPicked.connect(lambda: self.map.set_color_gradiant(
            [self._colorMin.color, self._colorMiddle.color, self._colorMax.color]))

    def on_mouse_moved(self, x, y, z):
        self._info_bar.setText(f'({x:d}, {y:d}) -> {z:.2f}')

    def fetch_params(self, prefs):
        prefs.ripple_color_min = self._colorMin.color
        prefs.ripple_color_middle = self._colorMiddle.color
        prefs.ripple_color_max = self._colorMax.color

    def load_params(self, prefs):
        self._colorMin.color = prefs.ripple_color_min
        self._colorMiddle.color = prefs.ripple_color_middle
        self._colorMax.color = prefs.ripple_color_max
        self.map.set_color_gradiant([self._colorMin.color, self._colorMiddle.color, self._colorMax.color])


class DialogProg(QtWidgets.QDialog):
    """ Dialog for displaying progress """

    def __init__(self, title, parent=None):
        super().__init__(parent)

        self.setWindowTitle(title)
        self.setMinimumWidth(300)

        self.progBar = QtWidgets.QProgressBar()
        self.progBar.setValue(0)
        self.progBar.setRange(0, 1)
        self.btnAbort = QtWidgets.QPushButton('Abort')

        thisLayout = QtWidgets.QVBoxLayout()
        thisLayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter)
        thisLayout.addWidget(self.progBar)
        thisLayout.addWidget(self.btnAbort)
        self.setLayout(thisLayout)

    def set_nt(self, n):
        """ Reset progress bar """
        self.progBar.setRange(0, n)
        self.progBar.setValue(0)

    def update_prog(self, n):
        self.progBar.setValue(n)


class ColorPicker(QtWidgets.QLabel):
    """ Pick color """

    colorPicked = QtCore.pyqtSignal()

    def __init__(self, dialogColor, color='#000000', parent=None):
        super().__init__(parent)

        self.setFixedWidth(30)
        self.setFixedHeight(20)
        self.setStyleSheet('background-color: {:s}; border: 1pt'.format(color))
        self._color = color
        self._dialog = dialogColor

    def mouseReleaseEvent(self, ev):

        qc = QtGui.QColor()
        qc.setNamedColor(self._color)
        self._dialog.setCurrentColor(qc)
        self._dialog.exec()
        if self._dialog.result() == QtWidgets.QDialog.DialogCode.Accepted:
            self.color = self._dialog.selectedColor().name()
            self.colorPicked.emit()

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color
        self.setStyleSheet('background-color: {:s}'.format(color))
        qc = QtGui.QColor()
        qc.setNamedColor(color)
        self._dialog.setCurrentColor(qc)


class _ThreadRender(QtCore.QThread):

    sig_complete = QtCore.pyqtSignal()
    sig_set_nt = QtCore.pyqtSignal(int)
    sig_update_prog = QtCore.pyqtSignal(int)
    sig_return_result = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hyper_res = None
        self._dim = None
        self._lambda_ = None
        self._damp = None
        self._pt_source_lambdas = None
        self._pt_source_amps = None
        self._pt_source_phases = None
        self._abort = False

    def setup(self, hyper_res, dim, damp, pt_source_lambdas, pt_source_amps, pt_source_phases):
        self._abort = False
        self._hyper_res = hyper_res
        self._dim = dim
        self._damp = damp
        self._pt_source_lambdas = pt_source_lambdas
        self._pt_source_amps = pt_source_amps
        self._pt_source_phases = pt_source_phases

    def abort(self):
        self._abort = True

    def run(self):
        wave_mat = np.zeros((self._dim * self._hyper_res, self._dim * self._hyper_res), dtype=complex)
        non_zero_indices = np.argwhere(self._pt_source_amps > 0)
        self.sig_set_nt.emit(len(non_zero_indices))
        c = 0
        for x0, y0 in non_zero_indices:
            if not self._abort:
                wave_mat += wave(self._hyper_res, self._dim, x0, y0, self._pt_source_amps[x0, y0],
                                 self._pt_source_phases[x0, y0], self._pt_source_lambdas[x0, y0], self._damp)
                c += 1
                self.sig_update_prog.emit(c)
            else:
                break
        img = np.real(wave_mat)
        self.sig_return_result.emit(img)
        self.sig_complete.emit()


if __name__ == '__main__':

    # fix the bug of bad scaling on screens of different DPI
    if platform.system() == 'Windows':
        if int(platform.release()) >= 8:
            ctypes.windll.shcore.SetProcessDpiAwareness(True)

    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont('Microsoft YaHei UI', 12))

    window = MainWin()
    window.show()

    sys.exit(app.exec())
