# USAGE
# To read and write back out to video:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 \
#	--output output/output_01.avi
#
# To read from webcam and write back out to disk:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
#	--output output/webcam_output.avi

# import the necessary packages
import UtilsIO
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import FPS
import numpy as np
import imutils
import dlib
import cv2

from roi_elements import RoiElements

line = (0, 0, 0, 0)
startPoint = False
endPoint = False

lineFlag = False

startCountFlag = False


def on_mouse(event, x, y, flags, params):
    global line, startPoint, endPoint

    # get mouse click
    if event == cv2.EVENT_LBUTTONDOWN:

        if startPoint == True and endPoint == True:
            startPoint = False
            endPoint = False
            line = []

        if startPoint == False:
            line = (x, y, 0, 0)
            startPoint = True
        elif endPoint == False:
            endPoint = True
            line = (line[0], line[1], x, y)


# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")

net = cv2.dnn.readNetFromCaffe("mobilenet_ssd/MobileNetSSD_deploy.prototxt",
                               "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")

# if a video path was not supplied, grab a reference to the webcam
print("[INFO] starting video stream...")

vs = cv2.VideoCapture(UtilsIO.SAMPLE_FILE_NAME)

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

initBB = None
roi_area = None
roi_elements = None

frame_size_w = 500
frame_size_h = 400

cv2.namedWindow('preview')

cv2.setMouseCallback('preview', on_mouse)

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=50, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
    # grab the next frame and handle if we are reading from either
    # VideoCapture or VideoStream
    ret, frame = vs.read()

    # resize the frame to have a maximum width of 500 pixels (the
    # less data we have, the faster we can process it), then convert
    # the frame from BGR to RGB for dlib
    frame = cv2.resize(frame, (frame_size_w, frame_size_h))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if lineFlag:
        if startPoint == True and endPoint == True:
            try:
                cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (255, 0, 255), 2)
            except:
                pass

    if initBB is not None and roi_elements is None or (initBB is not None and initBB != roi_elements.box):
        roi_elements = RoiElements(initBB)

    if initBB is not None:
        roi_area = roi_elements.getRoiArea(frame)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # initialize the current status along with our list of bounding
    # box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status = "Waiting"
    rects = []

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % 20 == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []
        if roi_area is not None:
            # convert the frame to a blob and pass the blob through the
            # network and obtain the detections
            blob = cv2.dnn.blobFromImage(roi_area, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()
        else:
            # convert the frame to a blob and pass the blob through the
            # network and obtain the detections
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by requiring a minimum
            # confidence
            if confidence > 0.4:
                # extract the index of the class label from the
                # detections list
                idx = int(detections[0, 0, i, 1])

                # if the class label is not a person, ignore it
                if CLASSES[idx] != "person":
                    continue

                # compute the (x, y)-coordinates of the bounding box
                # for the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                if roi_elements is not None:

                    left = startX + roi_elements.roi_area.coord_right
                    top = startY + roi_elements.roi_area.coord_bottom
                    right = endX + roi_elements.roi_area.coord_bottom
                    bottom = endY + roi_elements.roi_area.coord_right

                else:
                    left = startX
                    top = startY
                    right = endX
                    bottom = endY

                rect = dlib.rectangle(int(left), int(top),
                                      int(right), int(bottom))

                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                trackers.append(tracker)

    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    else:
        # loop over the trackers
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)

    if startCountFlag and roi_elements.line is not None:
        dy = roi_elements.calcMeanYDistance()
        dx = roi_elements.calcMeanXDistance()
        angle = roi_elements.calculateAngle()

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():

        if roi_elements is not None and roi_elements.line is not None:
            if roi_elements.checkCentroidInsideLine(centroid) == False:
                pass

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
            directionY = centroid[1] - np.mean(y)
            x = [c[0] for c in to.centroids]
            directionX = centroid[0] - np.mean(x)

            if not to.counted:

                if roi_elements is not None and roi_elements.line is not None:

                    if angle > 45 and directionX > 0 and centroid[0] > dx:
                        totalDown += 1
                        to.counted = True

                    elif angle < 45 and directionY > 0 and centroid[1] > dy:
                        totalDown += 1
                        to.counted = True

                elif (directionY > 0 and centroid[1] > frame_size_h // 2):
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
        ("Down", totalDown),
        ("Status", status),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if roi_elements != None:
        # roi area
        cv2.rectangle(frame, (roi_elements.roi_area.coord_left, roi_elements.roi_area.coord_top),
                      (roi_elements.roi_area.coord_right, roi_elements.roi_area.coord_bottom),
                      (0, 255, 0), 2)
    if roi_elements != None and roi_elements.line != None:
        cv2.line(frame, (roi_elements.line[0], roi_elements.line[1]),
                 (roi_elements.line[2], roi_elements.line[3]),
                 (255, 0, 255), 2)

    # show the output frame
    cv2.imshow("preview", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    if key == ord("s"):
        initBB = None
        initBB = cv2.selectROI("ROI_FRAME", frame, fromCenter=False,
                               showCrosshair=False)

    if key == ord("x"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        if lineFlag == False:
            lineFlag = True
        else:
            lineFlag = False

    if key == ord("b"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        if startCountFlag is False:
            startCountFlag = True
            if roi_elements is not None and roi_elements.line is None:
                roi_elements.setLine(line)
                totalDown = 0
        else:
            startCountFlag = False

    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# if we are not using a video file, stop the camera video stream
vs.stop()

# otherwise, release the video file pointer
vs.release()

# close any open windows
cv2.destroyAllWindows()
