OK = "ok"


class CameraObject:

    def __init__(self, camera_name, url, statusBtn):
        self.camera_name = camera_name
        self.url = url
        self.statusBtn = statusBtn
        self.location = ""

    def setUrl(self, url):
        self.url = url

    def setCameraName(self, camera_name):
        self.camera_name = camera_name

    def setStatusBtn(self, path):
        self.statusBtn = path

    def setLocation(self, location):
        self.location = location


def checkDataForConfigCameraFrame(camera_obj):
    if len(camera_obj.url) == 0 and len(camera_obj.camera_name) == 0:
        return "Camera name and url must be filled!"
    elif len(camera_obj.url) != 0 and len(camera_obj.camera_name) == 0:
        return "Camera name must be filled!"
    elif len(camera_obj.url) == 0 and len(camera_obj.camera_name) != 0:

        return "Url field must be filled!"
    else:
        return OK
