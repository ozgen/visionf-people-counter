import numpy as np
import tensorflow as tf
import cv2
import time

import UtilsIO
import config
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import dlib
import threading
import imutils
import os
from imutils.video import FPS
import requests


from roi_elements import RoiElements
videoName = 1
#detectVideoName = 1
startTime = time.time()

line = (0, 0, 0, 0)
startPoint = False
endPoint = False

#initBB = None
# kamera 7de sürekli duranları almamak için alan kısıtlaması
initBB = (0, 0, 498, 280)
roi_area = None
roi_elements = None
#totalDown = 0


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        # print("Elapsed Time:", end_time - start_time)

        im_height, im_width, _ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0, i, 0] * im_height),
                             int(boxes[0, i, 1] * im_width),
                             int(boxes[0, i, 2] * im_height),
                             int(boxes[0, i, 3] * im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

model_path = 'faster_rcnn_inception_v2/frozen_inference_graph.pb'
odapi = DetectorAPI(path_to_ckpt=model_path)


def videoDetection(camNumber, detectVideoName, totalDown):
    global  line, startPoint, endPoint, lineFlag, startCountFlag, odapi, initBB, roi_area, roi_elements

    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    totalFrames = 0
    W = None
    H = None

    frame_size_w = 500
    frame_size_h = 400

    model_path = 'faster_rcnn_inception_v2/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    newFolder = "not_evaluated/"+ str(camNumber) + "/"
    cap = cv2.VideoCapture(newFolder +str(detectVideoName) + ".avi")

    if int(detectVideoName) > 1 and os.path.exists(newFolder + str(detectVideoName - 1) + ".avi") == True:
        os.remove(newFolder + str(detectVideoName - 1) + ".avi")

    # start the frames per second throughput estimator
    fps = FPS().start()

    if (cap.isOpened() == False ):
        if(len(os.listdir("./not_evaluated/" + str(camNumber))) == 0):
            r = requests.post("http://localhost:3000/api/counter", data={"counter": totalDown, "channelId": "13"})
            totalDown = 0
            cap.release()
            cv2.destroyAllWindows()
        else:
            detectVideoName += 1
            videoDetection(camNumber, detectVideoName, totalDown)

    while cap.isOpened() == True:
        r, img = cap.read()

        if img is None:
            #if int(detectVideoName) > 1 and os.path.exists("not_evaluated/" + str(detectVideoName - 1) + ".avi") == True:
            #    os.remove("not_evaluated/" + str(detectVideoName - 1) + ".avi")
            if(detectVideoName % 2 == 0):
                r = requests.post("http://localhost:3000/api/counter", data={"counter": totalDown, "channelId": camNumber})
                print(r.status_code)
                totalDown = 0

            detectVideoName += 1
            cap.release()
            cv2.destroyAllWindows()
            videoDetection(camNumber, detectVideoName, totalDown)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (frame_size_w, frame_size_h))
        cv2.namedWindow(str(camNumber))

        if initBB is not None and roi_elements is None or (initBB is not None and initBB != roi_elements.box):
            roi_elements = RoiElements(initBB)

        if initBB is not None:
            roi_area = roi_elements.getRoiArea(img)

        if W is None or H is None:
            (H, W) = img.shape[:2]

        status = "Waiting"
        rects = []

        if totalFrames % 8 == 0:
            status = "Detecting"
            trackers = []
            if roi_area is not None:
                boxes, scores, classes, num = odapi.processFrame(roi_area)

            else:
                boxes, scores, classes, num = odapi.processFrame(img)

            # Visualization of the results of a detection.
            for i in range(len(boxes)):
                # Class 1 represents human
                if classes[i] == 1 and scores[i] > threshold:
                    box = boxes[i]

                    tracker = dlib.correlation_tracker()

                    if roi_elements is not None:

                        left = box[1] + roi_elements.roi_area.coord_right
                        top = box[0] + roi_elements.roi_area.coord_bottom
                        right = box[3] + roi_elements.roi_area.coord_right
                        bottom = box[2] + roi_elements.roi_area.coord_bottom
                        rect = dlib.rectangle(int(left), int(top),
                                              int(right), int(bottom))
                    else:
                        left = box[1]
                        top = box[0]
                        right = box[3]
                        bottom = box[2]
                        rect = dlib.rectangle(int(left), int(top),
                                              int(right), int(bottom))
                    tracker.start_track(rgb, rect)

                    trackers.append(tracker)

        else:
            for tracker in trackers:
                status = "Tracking"

                tracker.update(rgb)
                pos = tracker.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                rects.append((startX, startY, endX, endY))
                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

            #        cv2.line(img, (0, roi), (W, roi), (0, 255, 255), 2)

            objects = ct.update(rects)

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

                        if (directionY > 0 and centroid[1] > frame_size_h // 3):
                            totalDown += 1
                            to.counted = True

                # store the trackable object in our dictionary
                trackableObjects[objectID] = to

                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "ID {}".format(objectID)
                cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            # construct a tuple of information we will be displaying on the
            # frame
            info = [
                ("Down", totalDown),
                ("Status", status),
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(img, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # show the output frame
            cv2.imshow(str(camNumber), img)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            # increment the total number of frames processed thus far and
            # then update the FPS counter
        totalFrames += 1
        fps.update()

        # stop the timer and display FPS information
    fps.stop()
    # print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()


if __name__ == '__main__':
    #videoDetection()
    t0 = threading.Thread(target=videoDetection, args = (7,1,0))
    t0.start()
    t1 = threading.Thread(target=videoDetection, args = (13,1,0))
    t1.start()
