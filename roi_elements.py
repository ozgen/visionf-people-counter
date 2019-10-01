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
    line = (0, 0, 0, 0)

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
        return d

    def getRoiArea(self, img):
        return self.roi_area.getRoiArea(img)
