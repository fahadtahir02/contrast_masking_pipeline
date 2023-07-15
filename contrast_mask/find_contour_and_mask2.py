import cv2
import numpy as np


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
detector = cv2.aruco.ArucoDetector(dictionary)
parameters = cv2.aruco.DetectorParameters
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX




def filter_intensity(frame, min_intensity, max_intensity):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # min_intensity = 165
    # max_intensity = 170

    # Create a binary mask based on the intensity range
    mask = np.zeros_like(gray_frame)
    mask[np.logical_and(gray_frame >= min_intensity, gray_frame <= max_intensity)] = 255

    # Apply the mask to the original frame to obtain the filtered frame
    filtered_frame = cv2.bitwise_and(frame, frame, mask=mask)

    return filtered_frame



def mask_intensity(frame):
    min_intensity = 165
    max_intensity = 170

    # Filter the frame based on intensity range (e.g., minimum intensity of 165 and maximum intensity of 170)
    filtered_frame = filter_intensity(frame, min_intensity, max_intensity) # 165, 170 wokr

    # Convert the filtered frame to grayscale
    gray_filtered = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)

    # Find contours in the filtered frame
    contours, _ = cv2.findContours(gray_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = cv2.inRange(gray_filtered, min_intensity, max_intensity)







    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)


    # Dilate the masked image to enhance the contours
    kernel_size = 75  # Size of the kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_frame = cv2.dilate(mask, kernel)

    # Find contours in the mask
    contours, hierarchy = cv2.findContours(dilated_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    


    
    result = np.zeros_like(frame)

    # Draw the contours on the result image
    cv2.drawContours(result, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Apply the inverse of the mask to the original frame
    inverse_mask = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(frame, result, mask=inverse_mask)


    # Add the masked frame to the result image
    result = cv2.add(result, masked_frame)
    return result




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
    framed = mask_intensity(frame)
    new_trans_color = cv2.cvtColor(framed, cv2.COLOR_BGR2GRAY) #COLOR_BGR2GRAY
    new_trans = cv2.bitwise_not(new_trans_color)
    
    #clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16,16)) #DELETE reduces detectability
    #new_trans = clahe.apply(new_trans)
    new_trans = cv2.GaussianBlur(new_trans, (9, 9), 1)
    new_trans = cv2.adaptiveThreshold(new_trans, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 45,1)
    
    #_, new_trans = cv2.threshold(new_trans, 80, 255, cv2.THRESH_BINARY)
    
    
    detections = detection(new_trans)
    return detections








# Initialize the camera
camera = cv2.VideoCapture(0)  # Adjust the parameter (0 or 1) based on the camera index




while True:
    # Capture a frame from the camera
    ret, frame = camera.read()
    pre_proccessingg = pre_proccessing(frame)
    mask = mask_intensity(frame)
    
    # stack the initialized screens
    stacked = np.hstack((frame, mask, pre_proccessingg))
    

    cv2.imshow('frame Window', stacked)


    # Check for user input to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()

