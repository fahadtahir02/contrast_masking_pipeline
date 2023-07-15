import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Initialize the camera
camera = cv2.VideoCapture(0)  # Adjust the parameter (0 or 1) based on the camera index

# Initialize pixel intensity range
min_intensity = 0
max_intensity = 255

def update_range(val):
    # Update the pixel intensity range based on the slider values
    global min_intensity, max_intensity
    min_intensity = int(min_slider.val)
    max_intensity = int(max_slider.val)

# Create a blank figure and axis for the histogram plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Create sliders for the pixel intensity range
min_slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='lightgoldenrodyellow')
min_slider = Slider(min_slider_ax, 'Min Intensity', 0, 255, valinit=min_intensity, valstep=1)
max_slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor='lightgoldenrodyellow')
max_slider = Slider(max_slider_ax, 'Max Intensity', 0, 255, valinit=max_intensity, valstep=1)

# Link the update_range function to the slider update events
min_slider.on_changed(update_range)
max_slider.on_changed(update_range)

while True:
    # Capture a frame from the camera
    ret, frame = camera.read()

    # Crop the portion of the frame
    x = 800  # X-coordinate of the top-left corner of the cropped area
    y = 800  # Y-coordinate of the top-left corner of the cropped area
    width = 900  # Width of the cropped area
    height = 800  # Height of the cropped area
    cropped_frame = frame[y:y+height, x:x+width]

    # Convert the cropped frame to grayscale
    gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

    # Calculate the histogram
    histogram, bins = np.histogram(gray_frame.flatten(), 256, [0, 256])

    # Calculate the contrast
    contrast = np.std(gray_frame)

    # Create a binary mask based on the pixel intensity range
    mask = np.zeros_like(gray_frame)
    mask[np.logical_and(gray_frame >= min_intensity, gray_frame <= max_intensity)] = 255

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the frame
    cv2.drawContours(cropped_frame, contours, -1, (0, 255, 0), 2)

    # Plot the histogram
    ax.clear()
    ax.plot(histogram, color='black')
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram')

    # Find the valleys in the histogram (local minima)
    valleys = []
    for i in range(1, len(histogram) - 1):
        if histogram[i] < histogram[i-1] and histogram[i] < histogram[i+1]:
            valleys.append(i)
            print(valleys)

    # Display the frame and histogram
    cv2.imshow('Frame', cropped_frame)
    plt.pause(0.001)

    # Check for user input to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()
