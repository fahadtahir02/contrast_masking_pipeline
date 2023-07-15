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
    segmented_roi = cv2.bitwise_and(frame, frame, mask=mask)


    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16,16)) #Smootens out the image
    transformation = clahe.apply(mask)
    
    
    transformation = cv2.GaussianBlur(segmented_roi, (25, 25), 2)
    edges = cv2.Canny(segmented_roi, 50, 40)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    

    #Recreating mask that was created above to fit the now dilated contour
    #masked_dilated_contour = cv2.bitwise_and(dilated_contour, mask)
    #segmented_roi = cv2.bitwise_and(frame, frame, mask=masked_dilated_contour)
    
    transformation = cv2.drawContours(segmented_roi, contours, -1, (0, 255, 0), 2) # LAST NUMBER WAS 2 i changed it to 1 and it looks better 


    kernel_size = (5, 5)  # Size of the kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    iterations = 5  # Number of dilation iterations
    dilated_contour = cv2.dilate(transformation, kernel, iterations=iterations)

    

    

    


    return dilated_contour




cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    transformation_show = transformation(frame)
    cv2.imshow('frame Window', transformation_show)
    if cv2.waitKey(1) & 0XFF == ord('d'):
        break;
cap.release()
cv2.destroyAllWindows()



'''
PIPELINE SELF CRITIQUE:
- Only works withing the circular area of the light source provided by the camera. View in photo booth alongside vscode to get a better understanding. 

'''

'''
Use case for morphological operations: Dilation and erosion 

Removing noise
Isolation of individual elements and joining disparate elements in an image.
Finding of intensity bumps or holes in an image
'''





'''
Plan:
Step 1:Create a dilation of the masked and contoured current aruco tag
Step 2:mask the dilation 
Step 3:visualise both?
'''