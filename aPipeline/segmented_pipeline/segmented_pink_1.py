import cv2
from matplotlib import contour
from matplotlib.pyplot import draw
import numpy as np
import imutils
import skimage
from skimage import measure
import mahotas as mh



# the point of this copy is to utilize another way to calculate contour. opencv keeps changing their process and now it outputs contours in tuple. this creates an issue when using imutils.grab_contours becuase
# it expects the output to be in type list not tuple, i have tried numerous ways of converting it to list using another grab method instead of imutils but its all hopeless. Now i will be using contours = measure.find_contours(gray, level=0.5) from sklearn
def transformation(frame):
    #transformation = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    

    #Define the lower and upper thresholds for hue
    lower_hue = np.array([143, 106, 143])  # Lower hue threshold (approximate values)
    upper_hue = np.array([179, 255, 255])  # Upper hue threshold (approximate values)


    # Create a binary mask based on the hue thresholds
    mask = cv2.inRange(hsv_image, lower_hue, upper_hue)
    segmented_roi = cv2.bitwise_and(frame, frame, mask=mask)
    #clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16,16))
    #transformation = clahe.apply(mask)
    #transformation = cv2.GaussianBlur(transformation, (5, 5), 1)
    transformation = cv2.GaussianBlur(segmented_roi, (9, 9), 0)
    edges = cv2.Canny(segmented_roi, 10, 20)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    #contours = skimage.measure.find_contours(edges)
    #might have to change edges to segmented_roi

    #cnts = imutils.grab_contours(contours)
    #contours = contours[0] if len(contours) == 2 else contours[1]
    #contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    #print(contours)
    
    

    '''
    for c in contours:
        peri = (c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            print("screemCNT :", screenCnt)
            break
         '''   
    #WHY IS SCREENCNT not working
    #cc, rr = skimage.draw.polygon_perimeter(contours) # doesnt tell us coordinates rather asks for them???
    
    '''
    for contour in contours:
        # Convert contour to integer coordinates
        contour = np.round(contour).astype(int)

        # Extract the contour vertices
        vertices_r = contour[:, 0]
        vertices_c = contour[:, 1]

        # Draw the contour perimeter on the image
        rr, cc = draw.polygon_perimeter(vertices_r, vertices_c, shape=segmented_roi.shape, clip=True)
        segmented_roi[rr, cc] = 255  # Set pixel values to 255 (white)

    return transformation(segmented_roi)
    '''
    #transformation = cv2.drawContours(segmented_roi, contours, -1, (0, 255, 0), 2)
    transformation = cv2.drawContours(segmented_roi, contours, -1, (0, 255, 0), 2)
    transformation = cv2.GaussianBlur(segmented_roi, (9, 9), 0)
    
    #print("this is contour :", cv2.drawContours(segmented_roi, contours, -1, (0, 255, 0), 2))
    #x,y,z,w = cv2.drawContours(segmented_roi, contours, -1, (0, 255, 0), 2)
    #print('x :', x, 'y :', y, 'z :', z, 'w :', w) 
    
    


    #transformation = cv2.adaptiveThreshold(contours, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5,1)
    
    #_,transformation = cv2.threshold(transformation, 150, 255, cv2.THRESH_BINARY)
    

    return transformation




cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    transformation_show = transformation(frame)
    cv2.imshow('frame Window', transformation_show)
    if cv2.waitKey(1) & 0XFF == ord('d'):
        break;
cap.release()
cv2.destroyAllWindows()
