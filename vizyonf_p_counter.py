# import the necessary packages
import dlib

import imutils
import numpy as np
import cv2

# initialize the HOG descriptor/person detector
from imutils.video import FPS

import UtilsIO
import config
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

totalFrames = 0
totalDown = 0
totalUp = 0

fps = FPS().start()

'''HOG algoritmasının ilklendirmesi '''
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cv2.startWindowThread()
'''-----------------------------------------'''
# open webcam video stream or open a video
# cap = cv2.VideoCapture(UtilsIO.SAMPLE_FILE_NAME)


cap = cv2.VideoCapture(config.CONFIG_IP_CAM)
totatFrames = 0
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # resizing for faster detection
    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rects = []

    # 17 frame de bir capture ederek detection algoritmasını çalıştırıyor.
    if totatFrames % 10 == 0:

        # gri kullanarak detection algoritmasını az da hızlandırdık.
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # resim üzerinden insanları tespit et
        # dönüş tipi olarak insanların bulunduğu kutuları döner.

        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        status = "Detecting"
        trackers = []

        for box in boxes:
            # düzgün bir dikdörtgen için gerekli
            (startX, startY, endX, endY) = box.astype("int")
            # dikdörtgenin video üzerinde gösterir
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
            # sınırlayıcı kutu koordinatlarından bir dlib dikdörtgen nesnesi oluşturun ve
            # sonra dlib korelasyon izleyicisini başlatın
            tracker = dlib.correlation_tracker()
            try:
                tracker.start_track(rgb, rect)
                trackers.append(tracker)
            except:
                pass
    else:
        for tracker in trackers:
            # sistemimizin durumunu “Tracking” olarak ayarlamak
            status = "Tracking"

            # izleyiciyi güncelle ve güncellenmiş konumu yakala
            tracker.update(rgb)
            pos = tracker.get_position()

            # konum nesnesini açmak
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # sınırlayıcı kutu koordinatlarını dikdörtgenler listesine ekleyin
            rects.append((startX, startY, endX, endY))

        # çerçevenin ortasına yatay bir çizgi çizin
        # bir nesne bu çizgiyi geçtikten sonra 'yukarı' mı 'aşağı' mı hareket ettiklerini belirleyeceğiz
    (W, H) = frame.shape[:2]
    cv2.line(frame, (0, 200), (H, 200), (0, 255, 255), 2)

    # eski nesne centroidlerini yeni hesaplanan nesne centroidleriyle ilişkilendirmek için centroid izleyicisini kullanın
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object
                if direction < 0 and centroid[1] < H // 2:
                    totalUp += 1
                    to.counted = True

                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and centroid[1] > H // 2:
                    totalDown += 1
                    to.counted = True

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("Up", totalUp),
        ("Down", totalDown),
        ("Status", status),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # show the output frame
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1
    fps.update()

    # Display the resulting frame
    cv2.imshow('frame', frame)
    totatFrames = totatFrames + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
fps.stop()
cap.release()
# and release the output
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
