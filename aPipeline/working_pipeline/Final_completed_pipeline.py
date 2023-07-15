from inspect import Parameter
import cv2
from cv2 import waitKey
import sys
import numpy as np
from sklearn.model_selection import GridSearchCV
sys.path.append('/Users/fahadtahir/Library/Python/3.8/lib/python/site-packages')





#note theres a difference between an algorithm detecting the tags and the implementation of showing or drawing on video telling us what the algorithm sees. 
##in this case the algorithm is .ArucoDetector butt the detection pipeline will consist of another aruco module called .drawDetectedMarkers() which will show us what the pc is seeing.
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
detector = cv2.aruco.ArucoDetector(dictionary)
parameters = cv2.aruco.DetectorParameters
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX





def nothing (x):
    print(x)


def detection(frame):
    #Q: Do we need to detect before we can draw or does drawDetected..() do all that?

    
    #gives us what we need which is corner, ids and rejected
    corners, ids, rejected = detector.detectMarkers(frame)

    # Print the detected corners

    #print("Detected Corners:")
    #for corner_set in corners:
    #    for corner in corner_set:
    #        print(corner)
            
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) #max ieter, desired accuracy 
    winSize = (7, 7) # It provides a relatively small search neighborhood 
    zeroZone = (1, 1) #restricts the corner refinement process to a very local region around each corner

    

    refined_corners = []
    for corner_set in corners:
        refined = cv2.cornerSubPix(frame, corner_set, winSize, zeroZone, criteria)
        refined_corners.append(refined)
        for corner in refined_corners:
            print("Refined Corner: ", corner)
  
    

    # Calculate the refined corner locations

    visualizer = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    #We take that and input it in this module(.drawDet...(inserted here))
    visualizer = cv2.aruco.drawDetectedMarkers(visualizer, refined_corners, ids) 

    #return visualizer
    return visualizer



def transformation(frame):
    
    
    transformation = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    transformation = cv2.bitwise_not(transformation)
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16,16))
    transformation = clahe.apply(transformation)

    #flipped because the other side is opposite
    #transformation = cv2.flip(transformation, 1)
    

    
    # Flip vertically
    #flipped_vertically = cv2.flip(image, 0)
    #transformation = cv2.adaptiveThreshold(transformation, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 37,1)
    
    transformation = cv2.GaussianBlur(transformation, (21, 21), 0)
    #transformation = cv2.bilateralFilter(transformation, 9, 75, 75)
    #transformation = cv2.Canny(transformation, 10,20)
    #transformation = cv2.adaptiveThreshold(transformation, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 37,1)
    #transformation = cv2.equalizeHist(transformation)

    
    
    transformation = cv2.adaptiveThreshold(transformation, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 37,1)
    #keep this less than 67 because then there is too much noise


    # Perform corner detection using any corner detection algorithm (e.g., Harris corner detection)
    
    
    
    #corners = cv2.cornerHarris(transformation, blockSize=2, ksize=3, k=0.04)

    # Refine the corners using cv2.cornerSubPix()
    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #transformation = cv2.cornerSubPix(transformation, 6, (5,5), (-1, -1), criteria)


    #transformation = cv2.Canny(transformation, 150,300)
   
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



