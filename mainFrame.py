import PyQt5

import numpy as np
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

import os

import cv2

import config
import data_validation
from VideoPlayer import MainVideoPlayerWidget
from data_validation import CameraObject

FROM_MAIN, _ = loadUiType(os.path.join(os.path.dirname(__file__), "main.ui"))
FROM_SUB_EDIT, _ = loadUiType(os.path.join(os.path.dirname(__file__), "sub_edit.ui"))

imagePath = 'red.ico'


class SubEdit(QWidget, FROM_SUB_EDIT):

    def __init__(self, parent=None):
        super(SubEdit, self).__init__(parent)
        self.setupUi(self)

    def closeEvent(self, event):
        close = QMessageBox()
        close.setText("Are you sure?")
        close.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        close = close.exec()

        if close == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
    # let the window close


class QCustomQWidget(QWidget):

    def __init__(self, parent=None):
        super(QCustomQWidget, self).__init__(parent)
        self.textSecondQLabel = QtWidgets.QLabel()
        self.textSecondQLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.allQHBoxLayout = QtWidgets.QHBoxLayout()
        self.statusBtn = QtWidgets.QPushButton("")
        self.configBtn = QtWidgets.QPushButton("Configuration Edit")
        self.configBtn.setIcon(QIcon("config.ico"))
        self.allQHBoxLayout.addWidget(self.textSecondQLabel, 1)
        self.allQHBoxLayout.addWidget(self.statusBtn, 2)
        self.allQHBoxLayout.addWidget(self.configBtn, 3)
        self.allQHBoxLayout.addStretch()

        self.setLayout(self.allQHBoxLayout)

    def setTextSecond(self, text):
        self.textSecondQLabel.setText(str(text))

    def setStatusIcon(self, imgPath):
        self.statusBtn.setIcon(QIcon(imgPath))


class Main(QMainWindow, FROM_MAIN):
    main_widged = None
    cnt = 1
    item_list = []


    def __init__(self, parent=None):
        super(Main, self).__init__(parent)
        self.setupUi(self)
        self.addBtn.clicked.connect(self.addBtn_on_click)
        self.dellBtn1.clicked.connect(self.dellBtn1_on_click)
        self.sub_edit = SubEdit()
        self.camera_url.setText("videos/TownCentreXVID.avi")

    def addBtn_on_click(self):
        url = self.camera_url.text()
        camera_name = self.camera_name.text()
        camera_obj = CameraObject(camera_name, url, imagePath)
        ret_val = data_validation.checkDataForConfigCameraFrame(camera_obj)
        if ret_val == data_validation.OK:
            self.item_list.append(camera_obj)
            self.itemN = QListWidgetItem()
            self.custom_widget = QCustomQWidget()
            self.custom_widget.configBtn.clicked.connect(self.configBtn_on_click)

            self.custom_widget.setTextSecond(camera_name)
            self.custom_widget.setStatusIcon(imagePath)
            self.itemN.setSizeHint(self.custom_widget.sizeHint())
            self.itemN.setText(str(self.cnt))
            self.list_widget.addItem(self.itemN)
            self.list_widget.setItemWidget(self.itemN, self.custom_widget)
            self.cnt += 1



        else:

            QMessageBox.information(self, "Warning", ret_val)

    def dellBtn1_on_click(self):
        custom_widget_list = self.list_widget.selectedItems()
        if not custom_widget_list: return

        for cw in custom_widget_list:
            for i in range(len(self.item_list)):

                if cw.text() == str(i):
                    self.item_list.pop(i - 1)
        self.cnt = 1
        self.list_widget.clear()
        for co in self.item_list:
            self.itemN = QListWidgetItem()
            self.custom_widget = QCustomQWidget()
            self.custom_widget.setTextSecond(co.camera_name)
            self.itemN.setText(str(self.cnt))
            self.custom_widget.setStatusIcon(co.statusBtn)
            self.itemN.setSizeHint(self.custom_widget.sizeHint())
            self.list_widget.addItem(self.itemN)
            self.list_widget.setItemWidget(self.itemN, self.custom_widget)
            self.cnt += 1

    def configBtn_on_click(self):
        custom_widget_list = self.list_widget.selectedItems()
        if not custom_widget_list: return

        for cw in custom_widget_list:
            for i in range(len(self.item_list)):

                if cw.text() == str(i + 1):
                    co = self.item_list[i]
                    self.sub_edit.name.setText(co.camera_name)
                    self.sub_edit.url_name.setText(co.url)
                    if len(co.location) > 0:
                        self.sub_edit.location.setText(co.location)
                    self.video_player = MainVideoPlayerWidget(co.url)
                    self.sub_edit.imageScreen.addWidget(self.video_player)
                    self.sub_edit.roiBtn.clicked.connect(self.video_player.on_roi_click)
                    self.sub_edit.lineBtn.clicked.connect(self.video_player.on_line_click)

        self.sub_edit.show()
