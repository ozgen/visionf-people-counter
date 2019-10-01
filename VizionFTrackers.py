import cv2

CSRT_TRACKER ="csrt"
KFC_TRACKER ="kcf"
BOOSTING_TRACKER ="boosting"
MIL_TRACKER ="mil"
TLD_TRACKER ="tld"
MEDIANFLOW_TRACKER ="medianflow"
MOSSE_TRACKER ="mosse"



def createTracker(trackerName):
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
    tracker = OPENCV_OBJECT_TRACKERS[trackerName]()

    return tracker


def setFrameInfo(trackerName, success, fps, frame):
    (H, W) = frame.shape[:2]
    info = [
        ("Tracker",trackerName),
        ("Success", "Yes" if success else "No"),
        ("FPS", "{:.2f}".format(fps.fps())),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)





