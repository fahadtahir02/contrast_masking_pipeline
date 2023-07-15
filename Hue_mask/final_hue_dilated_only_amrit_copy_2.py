import fractions
from xml.dom.minidom import ProcessingInstruction
import cv2
from matplotlib import contour
import numpy as np
import imutils



dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
detector = cv2.aruco.ArucoDetector(dictionary)
parameters = cv2.aruco.DetectorParameters
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX


#This Function filters the frame to visualize clearly based on hue and darken everythiing else becuse it is insignificant
def hue(frame):
    #transformation = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    

    #Define the lower and upper thresholds for hue
    lower_hue = np.array([143, 106, 202])  # Lower hue threshold (approximate values)
    upper_hue = np.array([179, 255, 255])  # Upper hue threshold (approximate values)


    # Create a binary mask based on the hue thresholds
    mask = cv2.inRange(hsv_image, lower_hue, upper_hue)
    segmented_roi = cv2.bitwise_not(mask)


    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16,16)) #Smootens out the image
    transformation = clahe.apply(mask)
    
    
    transformation = cv2.GaussianBlur(segmented_roi, (25, 25), 2)
    edges = cv2.Canny(segmented_roi, 50, 40)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    

    transformation = cv2.drawContours(segmented_roi, contours, -1, (0, 255, 0), 2) # LAST NUMBER WAS 2 i changed it to 1 and it looks better 
    


    return transformation
    


    

'''
#After creating a mask and contouring from above hue function we take that mask and dialte it.
def dilate(frame):
    contours_passed = hue(frame) 
    kernel_size = 100  # Size of the kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    #iterations = 5  # Number of dilation iterations
    dilated_contour = cv2.dilate(contours_passed, kernel)

    # Create a mask for the dilated contour
    dilated_mask = cv2.inRange(dilated_contour, (0, 0, 0), (255, 255, 255))

    # Apply the dilated mask to the original frame to get the new region of interest
    new_roi = cv2.bitwise_and(frame, dilated_contour, mask=dilated_mask)

    
    # transformation = cv2.drawContours(new_roi, contours, -1, (0, 255, 0), 2) # LAST NUMBER WAS 2 i changed it to 1 and it looks better 
    

   
    return new_roi #or can return new_roi same thing?

'''

def erode(frame):
    dilated_passed = hue(frame) 
    kernel_size = 70  # Size of the kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    #iterations = 5  # Number of dilation iterations
    dilated_contour = cv2.erode(dilated_passed, kernel)

    # Create a mask for the dilated contour
    dilated_mask = cv2.inRange(dilated_contour, (0, 0, 0), (255, 255, 255))

    # Apply the dilated mask to the original frame to get the new region of interest
    new_roi = cv2.bitwise_and(frame, dilated_contour, mask=dilated_mask)

    
    # transformation = cv2.drawContours(new_roi, contours, -1, (0, 255, 0), 2) # LAST NUMBER WAS 2 i changed it to 1 and it looks better 
    

   
    return new_roi #or can return new_roi same thing?
#Our detection function
def detection(frame):
    corners, ids, rejected = detector.detectMarkers(frame)

    # Print the detected corners

    # verbose = corner in corner_set
    # if verbose:
    #     print('Detected Corner: ', corner)
            
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) #max ieter, desired accuracy 
    winSize = (7, 7) # It provides a relatively small search neighborhood 
    zeroZone = (1, 1) #restricts the corner refinement process to a very local region around each corner

    
    
    refined_corners = []
    for corner_set in corners:
        refined = cv2.cornerSubPix(frame, corner_set, winSize, zeroZone, criteria)
        refined_corners.append(refined)
        for corner in refined_corners:
            print("Refined Corner: ", corner)
    

    '''
    rejected_corners = []
    for corner_set in rejected:
        reject = cv2.cornerSubPix(frame, corner_set, winSize, zeroZone, criteria)
        rejected_corners.append(reject)
        #for corner in rejected_corners:
            #print("Refined Corner: ", rejected)
    '''

    # Calculate the refined corner locations

    visualizer = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    #We take that and input it in this module(.drawDet...(inserted here))
    visualizer = cv2.aruco.drawDetectedMarkers(visualizer, refined_corners, ids) 

    #return visualizer
    return visualizer

#After masking the dilated contour we then process that and try running detect on it to obtain our final results. 
def pre_proccessing(frame):
    framed = hue(frame)
    new_trans_color = cv2.cvtColor(framed, cv2.COLOR_BGR2GRAY) #COLOR_BGR2GRAY
    new_trans = cv2.bitwise_not(new_trans_color)
    
    #clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16,16)) #DELETE reduces detectability
    #new_trans = clahe.apply(new_trans)
    new_trans = cv2.GaussianBlur(new_trans, (9, 9), 1)
    #new_trans = cv2.adaptiveThreshold(new_trans, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25,1)
    
    _, new_trans = cv2.threshold(new_trans, 120, 255, cv2.THRESH_BINARY)
    
    
    detections = detection(new_trans)
    return detections


cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    #initializing ebery screen in every function: I used each function as a step. First screen step 1-Final.
    hue_show = hue(frame)
    #dilate_show = dilate(frame)
    
    #final_show = pre_proccessing(frame)
    
    # stack the initialized screens
    stacked = np.hstack((frame, hue_show))
    

    cv2.imshow('frame Window', hue_show)
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





'''Meeting Jun 27th - Thijs

dialate and then contour or vice versa order doestn matter - 
Reason to do above:
1) inside aruco tag isnt being dilated correctly 
2) Border is being dilated but theres a slight slits of non dilated contours that are very obvious and in the long run can lead to non detection. This
is why we will erode first so that the inside of our contour is filled so our aurco detector can see the contrast between the design. Then dilate so outside of our contour is
pushed out so that it provides context to aruco detector function


'''