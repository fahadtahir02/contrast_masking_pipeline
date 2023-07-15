import cv2
from cv2 import threshold
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
detector = cv2.aruco.ArucoDetector(dictionary)
parameters = cv2.aruco.DetectorParameters
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

# Initialize the camera
camera = cv2.VideoCapture(0)  # Adjust the parameter (0 or 1) based on the camera index

# Initialize neighborhood size for local contrast calculation
neighborhood_size = 5

def calculate_local_contrast(image):
    # Calculate the local standard deviation as a measure of contrast
    local_contrast = cv2.GaussianBlur(image, (neighborhood_size, neighborhood_size), 0)
    local_contrast = cv2.subtract(image, local_contrast)
    local_contrast = cv2.convertScaleAbs(local_contrast)

    return local_contrast

def update_neighborhood_size(val):
    # Update the neighborhood size based on the trackbar value
    global neighborhood_size
    neighborhood_size = cv2.getTrackbarPos('Neighborhood Size', 'Frame')
    if neighborhood_size % 2 == 0:
        neighborhood_size += 1

def update_min_contour_area(val):
    # Update the minimum contour area based on the trackbar value
    global min_contour_area
    min_contour_area = cv2.getTrackbarPos('Min Contour Area', 'Frame')

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

def contour_and_mask(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the local contrast
    local_contrast = calculate_local_contrast(gray_frame)

    # Apply threshold to create binary image based on contrast
    _, binary_image = cv2.threshold(local_contrast, 0, 255, cv2.THRESH_BINARY)
    

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area threshold
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            filtered_contours.append(contour)

    # Create a blank mask image
    mask = np.zeros_like(frame)

    # Draw contours on the mask
    cv2.drawContours(mask, filtered_contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Apply the mask to the original frame
    masked_frame = cv2.bitwise_and(frame, mask)

    return masked_frame


def update_adaptive_threshold(val):
    # Update the adaptive threshold value based on the trackbar value
    global adaptive_threshold_value
    adaptive_threshold_value = cv2.getTrackbarPos('Adaptive Threshold', 'Frame')
    
# Initialize adaptive threshold value (adjust as needed)
adaptive_threshold_value = 75
# Create a trackbar for adaptive threshold value
cv2.createTrackbar('Adaptive Threshold', 'Frame', adaptive_threshold_value, 255, update_adaptive_threshold)

def pre_processing(frame):
    framed = contour_and_mask(frame)
    new_trans_color = cv2.cvtColor(framed, cv2.COLOR_BGR2GRAY)
    new_trans = cv2.bitwise_not(new_trans_color)

    # Initialize adaptive threshold value (adjust as needed)
    adaptive_threshold_value = 75
    # Create a trackbar for adaptive threshold value



    new_trans = cv2.adaptiveThreshold(new_trans, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, adaptive_threshold_value, 1)

    detections = detection(new_trans)
    return detections

# Create a window for the stacked frame
cv2.namedWindow('Frame')

# Minimum contour area threshold (adjust as needed)
min_contour_area = 1000
# Create trackbars in the stacked window
cv2.createTrackbar('Neighborhood Size', 'Frame', neighborhood_size, 31, update_neighborhood_size)
cv2.createTrackbar('Min Contour Area', 'Frame', min_contour_area, 5000, update_min_contour_area)
cv2.createTrackbar('Adaptive Threshold', 'Frame', adaptive_threshold_value, 255, update_adaptive_threshold)

def plot_histogram(image):
    # Calculate the histogram of the image
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])

    # Plot the histogram
    plt.figure()
    plt.plot(hist, color='black')
    plt.fill_between(range(len(hist)), hist, color='gray')
    plt.xlim([0, 256])
    plt.ylim([0, np.max(hist) * 1.1])
    plt.title('Local Contrast Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()


def display_local_contrast(original_frame, local_contrast):
    # Create a stacked image with original frame and local contrast image
    stacked = np.hstack((original_frame, local_contrast))

    # Display the stacked image
    cv2.imshow('Local Contrast', stacked)

while True:
    # Capture a frame from the camera
    ret, frame = camera.read()

    processed = pre_processing(frame)

    # Call the contour_and_mask function to get the masked frame
    masked_frame = contour_and_mask(frame)

    after_local_contrast = calculate_local_contrast(frame)

    # Display the stacked frame
    cv2.imshow('Frame', np.hstack((frame, masked_frame, processed)))

    # Plot the histogram
    plot_histogram(frame)
    plot_histogram(after_local_contrast)
   

    # Check for user input to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()
