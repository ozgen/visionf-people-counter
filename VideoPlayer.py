import cv2

import numpy as np
from PyQt5.QtCore import QPoint, QSize, QRect, Qt
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets


class VideoPlayerCore(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture(path)

        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return
        read, data = self.camera.read()
        # data = self.camera.read()
        if read:
            img = cv2.resize(data, (500, 400))
            self.image_data.emit(img)
            self.img_data = img

    def stop_recording(self):
        if self.timer.isActive():
            self.timer.stop()

    def resume_recording(self):
        if self.timer.isActive() == False:
            self.timer.start(0, self)


class VideoPlayerWidget(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)
        self.begin = QtCore.QPoint()
        self.end = QtCore.QPoint()

    def image_data_slot(self, image_data):
        self.image_data = image_data
        if self.begin.x() != 0 and self.end.x() != 0:
            width = self.end.x() - self.begin.x()
            height = self.end.y() - self.begin.y()
            cv2.rectangle(image_data,
                          (self.begin.x(), self.begin.y()), (width, height), (0, 255, 255), 2)

        self.image = self.get_qimage(image_data)
        self.image_tmp = self.get_qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()


class ImgShowWidget(QtWidgets.QLabel):
    initBB = None
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.image = image
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)
        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = QPoint()
        pixmap01 = QtGui.QPixmap.fromImage(self.image)
        pixmap_image = QtGui.QPixmap(pixmap01)
        self.setPixmap(pixmap_image)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setScaledContents(True)
        self.setMinimumSize(1, 1)

    def mousePressEvent(self, event):

        if event.button() == Qt.LeftButton:
            self.origin = QPoint(event.pos())
            self.rubberBand.setGeometry(QRect(self.origin, QSize()))
            self.rubberBand.show()

    def mouseMoveEvent(self, event):

        if not self.origin.isNull():
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):

        if event.button() == Qt.LeftButton:
            self.rubberBand.show()
            x = self.origin.x()
            y = self.origin.y()


class MainVideoPlayerWidget(QtWidgets.QWidget):

    def __init__(self, path, parent=None):
        super().__init__(parent)

        self.image = QtGui.QImage()
        self.begin = QtCore.QPoint()
        self.end = QtCore.QPoint()

        self.video_player_widget = None
        self.video_player_widget = VideoPlayerWidget()

        self.video_player_core = VideoPlayerCore(path)

        image_data_slot = self.video_player_widget.image_data_slot
        self.video_player_core.image_data.connect(image_data_slot)
        self.ilayout = QtWidgets.QVBoxLayout()

        self.ilayout.addWidget(self.video_player_widget)
        self.setLayout(self.ilayout)
        self.video_player_core.start_recording()

    def on_roi_click(self):
        self.video_player_core.stop_recording()
        self.image = self.video_player_widget.image_tmp
        self.label = ImgShowWidget(self.image)
        self.ilayout.removeWidget(self.video_player_widget)
        self.ilayout.addWidget(self.label)

    def on_line_click(self):
        self.video_player_core.resume_recording()
        self.ilayout.removeWidget(self.label)
        self.label.deleteLater()
        self.ilayout.addWidget(self.video_player_widget)
