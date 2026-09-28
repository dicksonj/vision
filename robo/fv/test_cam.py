import cv2
print("cv2 file:   ", cv2.__file__)
print("cv2 version:", cv2.__version__)
build = cv2.getBuildInformation()
for line in build.split('\n'):
    if 'GStreamer' in line:
        print("GStreamer:  ", line.strip())