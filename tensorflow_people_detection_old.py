import numpy as np
import tensorflow as tf
import cv2
import time

from imutils.video import FPS

import UtilsIO
import config
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import dlib


from roi_elements import RoiElements


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
        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()


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


if __name__ == '__main__':
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    totalFrames = 0
    totalDown = 0
    W = None
    H = None
    roi = 250

    frame_size_w = 500
    frame_size_h = 400

    model_path = 'faster_rcnn_inception_v2/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7

    #cap = cv2.VideoCapture(UtilsIO.SAMPLE_FILE_NAME_2)
    cap = cv2.VideoCapture(config.CONFIG_IP_CAM)
    #cap = cv2.VideoCapture('videos/test.avi')

    # start the frames per second throughput estimator
    fps = FPS().start()
    cv2.namedWindow('preview')

    cv2.setMouseCallback('preview', on_mouse)

    while True:
        r, img = cap.read()
        if r:
            img = cv2.resize(img, (frame_size_w, frame_size_h))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            (H, W) = img.shape[:2]

        status = "Waiting"
        rects = []

        if totalFrames % 5 == 0:
            status = "Detecting"
            trackers = []

            boxes, scores, classes, num = odapi.processFrame(img)
            # Visualization of the results of a detection.
            for i in range(len(boxes)):
                # Class 1 represents human
                if classes[i] == 1 and scores[i] > threshold:
                    box = boxes[i]

                    tracker = dlib.correlation_tracker()

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
                    directionY = centroid[1] - np.mean(y)
                    x = [c[0] for c in to.centroids]
                    directionX = centroid[0] - np.mean(x)

                    if not to.counted:


                        if (directionY > 0 and centroid[1] > frame_size_h // 2):
                            print("*************")
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
            cv2.imshow("preview", img)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break



        totalFrames += 1
        fps.update()

        # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()
