import cv2
import pytesseract





#detector- “haarcascade_frontalface_default.xml” specifically for face detection
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'Haarcascade_frontalface_default.xml')
    
    

    




'''proccessing model- Here we are proccessing our raw footage to 
A)change raw to grayscale
B)Use a module of our detector "haar_cascade" that automatically that creates a square on the faces detected by haar.cascade
C)We created a for loop that runs this module on every face that is detected.
D)After running this function we return the transformed frame
'''


'''
#Question: What other transformations can we add to allow our computer to better recognize faces without messing with parameters of our alogorithm such as increasing
the size of minNeighbors which is currently 9 to reduce false positives but also missing the chance to detect an obvious.
Thoughts:

'''
def transformation(frame, haar_cascade):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame,(5,5),0)
    square_creator = haar_cascade.detectMultiScale(frame, 1.1, 9)
    for (x, y, w, h) in square_creator:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame
    
    

#camera activation and frame capturing
cap = cv2.VideoCapture(0)



#Creating while loop so we can constantly capture frames
while(True):
    ret, frame = cap.read()
    transformation_frame = transformation(frame, haar_cascade)

    cv2.imshow('frame', transformation_frame)
    if cv2.waitKey(1) & 0xFF == ord('d'):
        break
cap.release()
cv2.destroyAllWindows()

    

