from inspect import Parameter
import cv2
from cv2 import waitKey
import sys
import numpy as np
from sklearn.model_selection import GridSearchCV
sys.path.append('/Users/fahadtahir/Library/Python/3.8/lib/python/site-packages')


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
detector = cv2.aruco.ArucoDetector(dictionary)
parameters = cv2.aruco.DetectorParameters
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX






def detect_squares(frame):
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

    
    '''
    refined_corners = []
    for corner_set in corners:
        refined = cv2.cornerSubPix(frame, corner_set, winSize, zeroZone, criteria)
        refined_corners.append(refined)
        for corner in refined_corners:
            print("Refined Corner: ", corner)
    '''
    '''
    rejected_corners = []
    for corner_set in rejected:
        reject = cv2.cornerSubPix(frame, corner_set, winSize, zeroZone, criteria)
        rejected_corners.append(reject)
        for corner in rejected_corners:
            print("Refined Corner: ", rejected)
    '''
    # Iterate through the rejected frames
    for i, rejected_frame in enumerate(rejected):
        # Count the number of non-zero pixels in the rejected frame
        count = np.count_nonzero(rejected_frame)

        # If the current rejected frame has more non-zero pixels than the previous most apparent rejected frame
        if count > most_apparent_rejected_count:
            most_apparent_rejected_count = count
            most_apparent_rejected_frame = rejected_frame

    # Display the most apparent rejected frame
    #cv2.imshow('Most Apparent Rejected Frame', most_apparent_rejected_frame)

    # Calculate the refined corner locations

    visualizer = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    #We take that and input it in this module(.drawDet...(inserted here))
    rejected_visualizer = cv2.aruco.drawDetectedMarkers(visualizer, most_apparent_rejected_frame) 

    #return visualizer
    return rejected_visualizer



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


def transformation(rejected_visualizer):

    # Apply transformation operations to the input frame
    transformed_frame = cv2.cvtColor(rejected_visualizer, cv2.COLOR_BGR2GRAY)
    transformed_frame = cv2.bitwise_not(transformed_frame)
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16,16))
    transformed_frame = clahe.apply(transformed_frame)
    transformed_frame = cv2.flip(transformed_frame, 1)
    transformed_frame = cv2.GaussianBlur(transformed_frame, (21, 21), 0)
    transformed_frame = cv2.adaptiveThreshold(transformed_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 37, 1)
    detections = detection(transformation)
    return detections

# Rest of the code...




cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    transformation_show = transformation(frame)
    cv2.imshow('frame Window', transformation_show)
    if cv2.waitKey(1) & 0XFF == ord('d'):
        break;
cap.release()
cv2.destroyAllWindows()