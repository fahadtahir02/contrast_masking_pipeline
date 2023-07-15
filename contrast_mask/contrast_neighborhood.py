import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Initialize the camera
camera = cv2.VideoCapture(0)  # Adjust the parameter (0 or 1) based on the camera index

# Create a blank figure and axis for the histogram plot
fig, ax = plt.subplots()

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
    neighborhood_size = cv2.getTrackbarPos('Neighborhood Size', 'Controls')
    if neighborhood_size % 2 == 0:
        neighborhood_size += 1




# Create a window for the trackbar
cv2.namedWindow('Controls')
cv2.createTrackbar('Neighborhood Size', 'Controls', neighborhood_size, 31, update_neighborhood_size)

# Minimum contour area threshold (adjust as needed)
min_contour_area = 1000

while True:
    # Capture a frame from the camera
    ret, frame = camera.read()

    # Crop and convert the frame to grayscale
    # cropped_frame = frame[800:1600, 800:1700]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the local contrast
    local_contrast = calculate_local_contrast(gray_frame)

    # # Calculate the histogram
    # histogram, bins = np.histogram(local_contrast.flatten(), 256, [0, 256])

    # # Normalize the histogram
    # normalized_histogram = histogram / np.sum(histogram)

    # # Find the indices of the local maxima or peaks
    # peaks, _ = find_peaks(normalized_histogram)

    # # Plot the histogram values
    # ax.clear()
    # ax.plot(normalized_histogram, color='black')
    # ax.plot(peaks, normalized_histogram[peaks], 'ro')
    # ax.set_xlabel('Contrast')
    # ax.set_ylabel('Normalized Frequency')
    # ax.set_title('Normalized Histogram of Contrast with Peaks')

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

    contrast_show = calculate_local_contrast(frame)

    stacked = np.hstack((frame, contrast_show, masked_frame))

    # Display the masked frame
    cv2.imshow('Masked Frame', stacked)

    # Display the frame and histogram
    
    cv2.imshow('Frame', frame)
    plt.pause(0.001)

    # Check for user input to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()
