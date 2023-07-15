import cv2

# Initialize the camera
camera = cv2.VideoCapture(0)  # Adjust the parameter (0 or 1) based on the camera index

# Callback function for trackbar changes
def update_lab_values(val):
    global l_value, a_value, b_value
    l_value = cv2.getTrackbarPos('L', 'Controls')
    a_value = cv2.getTrackbarPos('A', 'Controls')
    b_value = cv2.getTrackbarPos('B', 'Controls')

# Create a window for the trackbars
cv2.namedWindow('Controls')
cv2.createTrackbar('L', 'Controls', 0, 255, update_lab_values)
cv2.createTrackbar('A', 'Controls', 0, 255, update_lab_values)
cv2.createTrackbar('B', 'Controls', 0, 255, update_lab_values)

while True:
    # Capture a frame from the camera
    ret, frame = camera.read()

    # Convert the frame to LAB color space
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Get the LAB values from the trackbars
    l_value = cv2.getTrackbarPos('L', 'Controls')
    a_value = cv2.getTrackbarPos('A', 'Controls')
    b_value = cv2.getTrackbarPos('B', 'Controls')

    # Update the LAB values in the LAB frame
    lab_frame[:, :, 0] = l_value
    lab_frame[:, :, 1] = a_value
    lab_frame[:, :, 2] = b_value

    # Convert the LAB frame back to BGR color space
    output_frame = cv2.cvtColor(lab_frame, cv2.COLOR_LAB2BGR)

    # Display the output frame
    cv2.imshow('Output', output_frame)

    # Check for user input to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()
