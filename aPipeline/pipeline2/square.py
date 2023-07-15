import cv2
import numpy as np

def detect_squares(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection (Canny)
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours and filter squares
    squares = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            squares.append(approx)

    return squares

# Open video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Detect squares
    detected_squares = detect_squares(frame)

    # Draw detected squares on the frame
    for square in detected_squares:
        cv2.drawContours(frame, [square], 0, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Square Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
