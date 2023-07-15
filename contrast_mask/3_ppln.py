import requests
import cv2 
from shadow_highlight_correction import correction
from time import time
import numpy as np

current_tag = None

#GOAL: 3 ppln check 
#import function from the main img p ppln
#from Fahads_ppln import Fdetection, Ftransformation - ✔ ( changed the functions) 

 
#import The_Main_imgP_ppln from function

#GOAL-1: saves the frames specifically which the img p ppln was able to detect (if the img p pln saw the id -
#  > save it ) - ✔.
#GOAL-2: First score test for the pipelines, - > return an image, and printes an id of the aruco tag - ✔.

#RESULT : test-score - 1 (basic) check's the ppln by return an an image with an id - ✔.

#UPDATES: Implemneted test-1 fro F ppln works -> return an id - ✔.

# GOAL-3: Implement Amrits ppln 


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters 
detector = cv2.aruco.ArucoDetector(dictionary)



def detect(frame):
    corners,ids,_ = detector.detectMarkers(frame)
    detected_markers = cv2.aruco.drawDetectedMarkers(frame, corners, ids)   
    return ids,detected_markers
    ''''
    if ids is None:
        detected_flag = False
    else:
        detected_flag = True
    return detected_markers, detected_flag
    '''

def process_frame0(frame):
     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
     clache = cv2.createCLAHE(clipLimit = 40) 
     frame_clache = clache.apply(gray)  
     th3 = cv2.adaptiveThreshold(frame_clache,125,cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,51,1) 
     blurred = cv2.GaussianBlur(th3,(21,21),0) 

     detected_frame, detected_flag = detect(frame,blurred)
     #if detected_flag:
        ### save this frame
        #cv2.imwrite("save_frame1", frame)
        #print("SAVED")
     return detected_frame 

#Just a process frame is returned 
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    clache = cv2.createCLAHE(clipLimit = 40) 
    frame_clache = clache.apply(gray)  
    th3 = cv2.adaptiveThreshold(frame_clache,125,cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,51,1) 
    blurred = cv2.GaussianBlur(th3,(21,21),0) 
    flipped = cv2.flip(blurred, 1)
    return flipped



# Fahads ppln Start  -----------------------------------------------------------------------------------------------------------
    
def detect_F(frame):
    corners, ids, _ = detector.detectMarkers(frame)
    refined_corners = []
    for corner_set in corners:
        refined = cv2.cornerSubPix(frame, corner_set, winSize=(7, 7), zeroZone=(-1, -1),
                                   criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        refined_corners.append(refined)
    gray = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    visualizer = cv2.aruco.drawDetectedMarkers(gray, refined_corners, ids)
    return ids,visualizer

def process_frame_F(frame):
    transformation = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    transformation = cv2.bitwise_not(transformation)
    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16, 16))
    transformation = clahe.apply(transformation)
    transformation = cv2.GaussianBlur(transformation, (21, 21), 0)
    _, transformation = cv2.threshold(transformation, 150, 255, cv2.THRESH_BINARY)
    flipped = cv2.flip(transformation, 1)
    return flipped

# Fahads ppln End -----------------------------------------------------------------------------------------------------------





#Amrit's ppln Start -----------------------------------------------------------------------------------------------------------------------------

THRESHOLD_PX_VAL = 100
CLIP_LIMIT = 20.0

clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=(5, 5))


def blur(image, kernel):
    return cv2.blur(image, kernel)
def threshold(image, px_val):
    # ret, thresholded = cv2.threshold(image, px_val, 255, cv2.THRESH_TOZERO)
    # ret, thresholded = cv2.threshold(thresholded, px_val, 1, cv2.THRESH_TOZERO_INV)
    thresholded = cv2.adaptiveThreshold(image, 
                                        255, 
                                        cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY, 
                                        89, 
                                        2)
    return thresholded

def invert(img):
    image_not = cv2.bitwise_not(img)
    return image_not


def contrast(image, clahe):
    clahe_out = clahe.apply(image)
    return clahe_out

def A_process_frame(img, calibration_frame = None):
    # img_undistorted = undistort(img)
    img_corrected = correction(img, 0, 0, 0, 0.6, 0.6, 30, .3)

    img_gray = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2GRAY)
    if calibration_frame is not None:
        img_norm = img_gray - calibration_frame
    else:
        img_norm = img_gray

    img_contrast_enhanced = contrast(img_norm, clahe)
    img_blurred = blur(img_contrast_enhanced, (5, 5))
    img_thresholded = threshold(img_blurred, THRESHOLD_PX_VAL)
    
    # log_variable_clip_limit(img_contrast_enhanced, img_thresholded)
    # img_inverted = invert(img_blurred)

    #img_detected = A_detect(img_thresholded, img, draw_rejected=False)

    # img_segmented = segment_image(img_blurred)
    flipped = cv2.flip(img_thresholded, 1)
    return flipped

def drawMarkers(img, corners, ids, borderColor=(255,0,0), thickness=25):
    if ids == []:
        ids = ["R"] * 100
    for i, corner in enumerate(corners):
        if ids[i] == 17:
            continue
        corner = corner.astype(int)
        cv2.line(img, (corner[0][0][0], corner[0][0][1]), (corner[0][1][0], corner[0][1][1]), borderColor, thickness)
        cv2.line(img, (corner[0][1][0], corner[0][1][1]), (corner[0][2][0], corner[0][2][1]), borderColor, thickness)
        cv2.line(img, (corner[0][2][0], corner[0][2][1]), (corner[0][3][0], corner[0][3][1]), borderColor, thickness)
        cv2.line(img, (corner[0][3][0], corner[0][3][1]), (corner[0][0][0], corner[0][0][1]), borderColor, thickness)
        cv2.putText(img, str(ids[i]), (corner[0][0][0], corner[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5, cv2.LINE_AA)
    return img



def A_detect(image, draw_rejected = False):
    (corners, ids, rejected) =  detector.detectMarkers(image)
    (corners_inv, ids_inv, rejected_inv) =  detector.detectMarkers(invert(image))
    (corners_hflip, ids_hflip, _) = detector.detectMarkers(invert(cv2.flip(image,1)))

    back_to_color = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

    if draw_rejected:
        detected = drawMarkers(back_to_color.copy(), rejected, [], borderColor=(255, 0, 0))
        detected = drawMarkers(detected.copy(), rejected_inv, [], borderColor=(255, 0, 0))
        detected = drawMarkers(detected.copy(), corners, ids, borderColor=(83, 235, 52))
        detected = drawMarkers(detected.copy(), corners_inv, ids_inv, borderColor=(83, 235, 52))
    else:
        detected = drawMarkers(back_to_color.copy(), corners, ids, borderColor=(83, 235, 52))
        detected = drawMarkers(detected.copy(), corners_inv, ids_inv, borderColor=(83, 235, 52))
        detected = drawMarkers(cv2.flip(detected.copy(), 1), corners_hflip, ids_hflip, borderColor=(83, 235, 52))


    return ids_hflip, detected
    





# End of the Amrit_ppln ------------------------------------------------------------------------------------------------------------------------------------


# For a first check test of pipeline (score) - ppln_check
# I need a frame processing fucntion from an image processing pipeline(A,F,GridSearcg genereted ones), and a raw image of the tG 
# return id and image 
def ppln_check(img, process_frame,detect):
 p_image = process_frame(img)
 ids, img = detect(p_image)
 return ids,img

def A_ppln_check(img, A_process_frame):
    p_image = A_process_frame(img)
    ids, detected =  A_detect(p_image)
 
    return ids, detected



    


cap = cv2.VideoCapture(0)

#from Fahads_ppln import Fdetection, Ftransformation

while(True):
    
    #Basic Check for the ppln, and returns the image of the id   
    img = cv2.imread('id_3_passed.jpg')
    id, img_check = A_ppln_check(img, A_process_frame)
    cv2.imshow('img_check',img_check)

    if cv2.waitKey(1000) & 0xFF == ord('x'):
        #cv2.imwrite('img_check.jpg', img_check)
        print('Detect id is:', id)
        break
     
     
cap.release()
cv2.destroyAllWindows()






