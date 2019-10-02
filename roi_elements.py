from math import atan2, pi

import numpy as np


class RoiArea:
    coord_left = 0
    coord_top = 0
    coord_right = 0
    coord_bottom = 0

    def __init__(self, box):
        (a, b, c, d) = box
        self.coord_left = a + c
        self.coord_top = b + d
        self.coord_right = a
        self.coord_bottom = b

    # params img is a frame that comes from cv2.VideoCapture function
    def getRoiArea(self, img):
        return img[self.coord_bottom: self.coord_top, self.coord_right:self.coord_left]


class RoiElements:
    # initBB values
    box = None
    # line to count
    line = None

    # roi area is upper class
    roi_area = None

    def __init__(self, box):
        self.box = box
        self.roi_area = RoiArea(self.box)

    def setLine(self, line):
        self.line = line

    def calculateDistance(self, centroid):
        p1 = np.array([self.line[0], self.line[1]])
        p2 = np.array([self.line[2], self.line[3]])
        (a, b) = centroid
        p3 = np.array([a, b])
        d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
        return int(d)

    def getRoiArea(self, img):
        return self.roi_area.getRoiArea(img)

    def checkRoiDistance(self, centroid):
        mean_y = int((self.line[1] + self.line[3]) / 2)
        (a, b) = centroid
        return b > mean_y

    def calcMeanYDistance(self):
        return int((self.line[1] + self.line[3]) / 2)

    def calcMeanXDistance(self):
        return int((self.line[0] + self.line[2]) / 2)

    def calculateAngle(self, centroid):
        # point 1
        x1 = self.line[0]
        y1 = self.line[1]

        # point 2
        x2 = self.line[2]
        y2 = self.line[3]

        deltax = x2 - x1
        deltay = y2 - y1

        angle_rad = atan2(deltay, deltax)
        angle_deg = abs(angle_rad * 180.0 / pi) % 90

        return int(angle_deg)
