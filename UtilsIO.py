import cv2
import os
import datetime

SAMPLE_FILE_NAME = "videos/example_01.mp4"
SAMPLE_FILE_NAME_2 = "videos/TownCentreXVID.avi"
TRAINED_MODEL_TENSORFLOW = "faster_rcnn_inception_v2/frozen_inference_graph.pb"
IMAGE_PATH= "images"

def saveImages(IPCam, imageNumber, frame):
    print("save class")
    date = datetime.date.today()
    camNumber = IPCam.split("=")[1]
    camNumber = camNumber.split("&")[0]
    dirnameCam= IMAGE_PATH + "/" + camNumber
    dirname= dirnameCam + "/" + str(date)
    if not os.path.exists(IMAGE_PATH):
        os.mkdir(IMAGE_PATH)

    if not os.path.exists(dirnameCam):
        os.mkdir(dirnameCam)

    if not os.path.exists(dirname):
            os.mkdir(dirname)

    cv2.imwrite(dirname + "/image_" + str(date) + "_" + str(imageNumber) + ".jpg", frame)
