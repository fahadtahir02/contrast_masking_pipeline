import cv2
from cv2 import waitKey
import sys
from sklearn.model_selection import GridSearchCV
sys.path.append('/Users/fahadtahir/Library/Python/3.8/lib/python/site-packages')





#note theres a difference between an algorithm detecting the tags and the implementation of showing or drawing on video telling us what the algorithm sees. 
##in this case the algorithm is .ArucoDetector butt the detection pipeline will consist of another aruco module called .drawDetectedMarkers() which will show us what the pc is seeing.
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
detector = cv2.aruco.ArucoDetector(dictionary)
parameters = cv2.aruco.DetectorParameters


def nothing (x):
    print(x)


def detection(frame):
    #Q: Do we need to detect before we can draw or does drawDetected..() do all that?


    #gives us what we need which is corner, ids and rejected
    corners, ids, rejected = detector.detectMarkers(frame)

    #We take that and input it in this module(.drawDet...(inserted here))
    visualizer = cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor = (0, 255, 0)) 


    #return visualizer
    return visualizer



def transformation(frame):
    transformation = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    transformation = cv2.bitwise_not(transformation)
    transformation = cv2.GaussianBlur(transformation, (3, 3), 2)

    transformation = cv2.adaptiveThreshold(transformation, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 37,1)
    #ransformation = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8))
    #transformation = cv2.GaussianBlur(transformation, (3, 3), 2)
    _,transformation = cv2.threshold(transformation, 150, 255, cv2.THRESH_BINARY)
    detections = detection(transformation)
    return detections

'''
def canny(frame):
    transformation = transformation(frame)
    transformation = cv2.Canny(transformation, 100,40)
    return transformation
'''

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    transformation_show = transformation(frame)
    cv2.imshow('frame Window', transformation_show)
    if cv2.waitKey(1) & 0XFF == ord('d'):
        break;
cap.release()
cv2.destroyAllWindows()