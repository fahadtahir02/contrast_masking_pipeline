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
    segmented_roi = cv2.bitwise_and(frame, frame, mask=mask)


    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16,16)) #Smootens out the image
    transformation = clahe.apply(mask)
    
    
    transformation = cv2.GaussianBlur(segmented_roi, (25, 25), 2)
    edges = cv2.Canny(segmented_roi, 50, 40)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    

    #Recreating mask that was created above to fit the now dilated contour
    #masked_dilated_contour = cv2.bitwise_and(dilated_contour, mask)
    #segmented_roi = cv2.bitwise_and(frame, frame, mask=masked_dilated_contour)
    
    transformation = cv2.drawContours(segmented_roi, contours, -1, (0, 0, 0), 2) # LAST NUMBER WAS 2 i changed it to 1 and it looks better 


    return transformation







'''CURRENTLY REPLACED BY morphological_operations
#After creating a mask and contouring from above hue function we take that mask and dialte it.
def dilate(frame):
    contours_passed = hue(frame)
    kernel_size = 70  # Size of the kernel
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

#Our detection function
def detection(frame):
    corners, ids, rejected = detector.detectMarkers(frame)

    # Print the detected corners
    
    # print("Detected Corners:")
    # for corner_set in corners:
    #     for corner in corner_set:
    #         print(corner)
            
    
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

'''
#LOGIC: For morphological_operations
# Pass contours from hue function into here and dilate those contours and create a mask of the dilation 
# Apply that mask onto original frame called frame 
# Pass contours fromn hue function into here and erode those contours and createa a mask of the erosion 
# Apply that eroded mask onto DILATED FRAME thereby combining both and returning it via latest roi
def morphological_operation(frame):
    contours_passed = hue(frame)
    kernel_size = 100
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Perform both erosion and dilation
    eroded_contour = cv2.erode(contours_passed, kernel)
    dilated_contour = cv2.dilate(eroded_contour, kernel)
    eroded_mask = cv2.inRange(eroded_contour, (0, 0, 0), (255, 255, 255))


    # Create a mask for the dilated contour
    dilated_mask = cv2.inRange(dilated_contour, (0, 0, 0), (255, 255, 255))
    
    # Apply the dilated mask to the original frame to get the new region of interest
    new_roi = cv2.bitwise_and(frame, dilated_contour, mask=dilated_mask)

    combined_roi = cv2.bitwise_and(new_roi, eroded_contour, mask = eroded_mask)
    
    return combined_roi
'''  

def morphological_operation(frame):
    contours_passed = hue(frame)
    kernel_size = 150
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Perform erosion and dilation
    eroded_contour = cv2.erode(contours_passed, kernel)
    dilated_contour = cv2.dilate(contours_passed, kernel)

    # Create a mask representing the union of erosion and dilation basically show regions effected by erosion or dilation so show both vs bitwise_and = show only area impacted by both ONLY. 
    mask = cv2.bitwise_or(eroded_contour, dilated_contour)
    return mask



#After masking the dilated contour we then process that and try running detect on it to obtain our final results. 
def pre_proccessing(frame):
    frame = hue(frame)
    mask = morphological_operation(frame)

    #new_trans = cv2.drawContours(mask, frame, -1, (0, 0, 0), 2)

    # new_trans = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    # new_trans = cv2.bitwise_not(new_trans)
    # clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16,16)) #Smootens out the image
    # new_trans = clahe.apply(new_trans)
    # new_trans = cv2.GaussianBlur(new_trans, (9, 9), 1)
    # _, new_trans = cv2.threshold(new_trans, 100, 255, cv2.THRESH_BINARY)
    #new_trans = cv2.adaptiveThreshold(new_trans, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 37,1)
    #detections = detection(new_trans)
    return mask


cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    #initializing ebery screen in every function: I used each function as a step. First screen step 1-Final.
    hue_show = hue(frame)
    morphological_show = morphological_operation(frame)
    final_show = pre_proccessing(frame)
    
    # stack the initialized screens
    stacked = np.hstack((hue_show, morphological_show, final_show))
    

    cv2.imshow('frame Window', stacked)
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



'''
Jun 27th- 6:09

TO DO::TAkE the mask of morphological_show and return that on the orifinal frame from hue()


'''