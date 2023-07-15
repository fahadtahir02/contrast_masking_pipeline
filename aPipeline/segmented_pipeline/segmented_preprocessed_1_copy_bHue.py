from xml.dom.minidom import ProcessingInstruction
import cv2
from matplotlib import contour
import numpy as np
import imutils

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
detector = cv2.aruco.ArucoDetector(dictionary)
parameters = cv2.aruco.DetectorParameters
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

def transformation(frame):
    #transformation = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    

    #Define the lower and upper thresholds for hue
    lower_hue = np.array([143, 106, 202])  # Lower hue threshold (approximate values)
    upper_hue = np.array([179, 255, 255])  # Upper hue threshold (approximate values)


    # Create a binary mask based on the hue thresholds
    mask = cv2.inRange(hsv_image, lower_hue, upper_hue)
    inverted_mask = cv2.bitwise_not(mask)
    segmented_roi = cv2.bitwise_and(frame, frame, mask=mask)
    #clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16,16))
    #transformation = clahe.apply(mask)
    #transformation = cv2.GaussianBlur(transformation, (5, 5), 1)
    transformation = cv2.GaussianBlur(segmented_roi, (25, 25), 2)
    edges = cv2.Canny(segmented_roi, 50, 40)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #contours = contours[0] if len(contours) == 2 else contours[1]
    #contours = imutils.grab_contours(contours)
    #print(type(contours))

    transformation = cv2.drawContours(segmented_roi, contours, -1, (0, 255, 0), 2) # LAST NUMBER WAS 2 i changed it to 1 and it looks better 

    

    '''
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    #for c in contours:
    #    first_corner = contours[0][0]
    #    second_point = contours[1][0]
     
    while len(contours): #NOTE: Still need to calculate area of these points and filter for the largest area detected. 
        if len(contours) == 4:
                first_point = contours[0][0]
                second_point = contours[1][0]
                third_point = contours[2][0]
                fourth_point = contours[3][0]
        
    
    src_points = np.array(first_point, second_point, third_point, fourth_point, dtype=np.float32)
    dst_points = np.array([[[0,0],[300,0],[0,300],[300,300]]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    transformation = cv2.warpPerspective(contours,M,(300,300))
    
    print("fourth point")
        
    '''
    #print(type(contours))
    #print(contours[:1])
    #cnts = imutils.grab_contours(contours)
    #contours = contours[0] if len(contours) == 2 else contours[1]
    #contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    #print(contours)
    
    

    '''
    for c in contours:
        peri = (c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            print("screemCNT :", screenCnt)
            break
         '''   
    #WHY IS SCREENCNT not working
    #NOWIMP: transformation = cv2.drawContours(segmented_roi, contours, -1, (0, 255, 0), 2)
    #IMP: transformation = cv2.drawContours(segmented_roi, contours, -1, (0, 255, 0), 2)
    
    
    #print("this is contour :", cv2.drawContours(segmented_roi, contours, -1, (0, 255, 0), 2))
    #x,y,z,w = cv2.drawContours(segmented_roi, contours, -1, (0, 255, 0), 2)
    #print('x :', x, 'y :', y, 'z :', z, 'w :', w) 
    
    


    #transformation = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5,1)
    
    #_,transformation = cv2.threshold(transformation, 150, 255, cv2.THRESH_BINARY)
    

    return transformation


'''

def prossessing(transformation):
    new_trans = cv2.cvtColor(transformation, cv2.COLOR_GRAY2BGR) #PASS THIS AS IS AND SEE HOW IT LOOKS WITHOUT GRAY!!!!!! FIRST TASK 2nd task read notes in bottom of one of these files.
    new_trans = cv2.bitwise_not(new_trans)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) #max ieter, desired accuracy 
    winSize = (7, 7) # It provides a relatively small search neighborhood 
    zeroZone = (1, 1) #restricts the corner refinement process to a very local region around each corner

    corners, ids, rejected = detector.detectMarkers(frame)

    
    refined_corners = []
    for corner_set in corners:
        refined = cv2.cornerSubPix(transformation, corner_set, winSize, zeroZone, criteria)
        refined_corners.append(refined)
        for corner in refined_corners:
            print("Refined Corner: ", corner)
    
    
    #new_trans = cv2.bitwise_not(new_trans)
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16,16))
    new_trans = clahe.apply(new_trans)
    new_trans = cv2.adaptiveThreshold(new_trans, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 37,1)
    _,new_trans = cv2.threshold(new_trans, 150, 255, cv2.THRESH_BINARY)
    

    visualizer = cv2.aruco.drawDetectedMarkers(new_trans, corners, ids)
    return visualizer

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




#Issue: To be able to use cv2.getPerspectiveTransform() i need to give it a 4x2 matrix which holds the coordinates of all 4 corners of the aruco tag. 3 of which need to be collinear
#How to get corners: cv2.findContours() will give us the coordinates/corners : NOTE DOCUMENTATION STATES: finding contours is like finding white object from black background. So remember, object to be found should be white and background should be black.  hoever false because contour currently works

#Documentation states: Contours is a Python list of all the contours in the image. Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object. NOTE: false i printed datatype of contour and it is tuple
#NOTE: This is what cv.CHAIN_APPROX_SIMPLE does. It removes all redundant points and compresses the contour, thereby saving memory. gives us only 4 points in contours.
#^so im printing contours but im getting tuples and im not sure exactly what the tuples are referring to meaning why are there so many tuples i thought chain_approx only returned 4? IM assuming since this is active camera its constantly changing? 
#^I guess this is where imutils.grab_contours(contours) comes in it supposedly grabs the coordinates. But I know it needs to be type list so it can grab but because we are returned type tuple as contour we cant use mutils- please confirm b4 amrit. ERROR: in 
#in grab_contours raise Exception(("Contours tuple must have length 2 or 3

#UPDATE: Scrap the idea of transformations they are not neccessary and impede on the detection algorithmns domain. 




#LEAVING THIS AS IS ONLY THING IS UNCOMMENT new_trans = cv2.bitwise_not(new_trans) in processing function
#If constant crash occurs comment out refined corner loop

#Summary: Detection not occurring because corner are not being detected because erosion of contour like we predicted would happen
'''
solution keep in mind the Union and also remember we have to make the focus of our transformations to be that erosioned contour. But i believe
this is already the case it just so happens that adaptive threshold makes everuthing else visible to us as well 


Things to note: is there noise within the erosioned contour area (aruco tag)? 
Also lets not convert to gray scale in process function? what happens then?


'''
