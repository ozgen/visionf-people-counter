import numpy as np
import cv2
import time
import config
import threading
import imutils
import os

#videoName = 1

def videoRecording(ipCam, camNumber, videoName):
    #global videoName
    cap = cv2.VideoCapture(ipCam)
    if (cap.isOpened() == False):
        print("Unable to read camera")

    startTime = time.time()
    frameWidth = int(cap.get(3))
    frameHeight = int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    if(os.path.isdir("./not_evaluated") == False ): os.mkdir("not_evaluated")
    newFolder = "./not_evaluated/" + str(camNumber);
    if (os.path.isdir(newFolder) == False): os.mkdir(newFolder)

    out = cv2.VideoWriter(newFolder +"/" + str(videoName) + ".avi", fourcc, 20.0, (frameWidth, frameHeight))

    while (True):
        ret, frame = cap.read()
        if ret == True:
            out.write(frame)
            frame = imutils.resize(frame, width=640, height=480)
            cv2.imshow(str(camNumber),frame)
            #if camNumber == 7: cv2.imshow("7",frame)
            #else: cv2.imshow("13",frame)

            if(int(time.time() - startTime) > 60):
                out.release()
                cap.release()
                cv2.destroyWindow("frame")
                videoName += 1
                videoRecording(ipCam, camNumber, videoName)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    #videoRecording()
    t1 = threading.Thread(target=videoRecording, args = (config.CONFIG_IP_CAM_7,7,1))
    t1.start()
    t2 = threading.Thread(target=videoRecording, args = (config.CONFIG_IP_CAM_13,13,1))
    t2.start()
