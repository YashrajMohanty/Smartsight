import cv2
import numpy as np

# Test videos/LA Walk Park.mp4
# Test videos/LA Walk Outdoor.mp4
# Test videos/LA Walk Indoor.mp4

def roi_threshold(frame_roi):
    frame_roi_hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV) # BGR to HSV

    roi_r_mean = np.mean(frame_roi[:,:,2].ravel())
    roi_g_mean = np.mean(frame_roi[:,:,1].ravel())
    roi_b_mean = np.mean(frame_roi[:,:,0].ravel())

    roi_s_mean = np.mean(frame_roi_hsv[:,:,1].ravel())

    roi_r_thresh = int(roi_r_mean - 40)
    roi_g_thresh = int(roi_g_mean - 30)
    roi_b_thresh = int(np.min([roi_b_mean+60,255]))
    roi_s_thresh = int(np.min([roi_s_mean+50,255]))
    return (roi_r_thresh, roi_g_thresh, roi_b_thresh, roi_s_thresh)

capture = cv2.VideoCapture('Test videos/LA Walk Park.mp4')
while capture.isOpened():
    ret, frame = capture.read()

    cv2.imshow('Frame', frame)

    frame_blur = cv2.medianBlur(frame,5)
    frame_blur = cv2.GaussianBlur(frame_blur, (5,5), cv2.BORDER_DEFAULT)
    hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV) # BGR to HSV

    frame_roi = frame[200:450, 40:600,:]

    roi_thresh = roi_threshold(frame_roi)

    frame[frame_blur[:,:,2] < roi_thresh[0]] = 0
    frame[frame_blur[:,:,1] < roi_thresh[1]] = 0
    frame[frame[:,:,0] > roi_thresh[2]] = 0
    frame[hsv[:,:,1] > roi_thresh[3]] = 0
    frame[frame[:,:,0] > 0] = 255 

    kernel = np.ones((7,7), dtype = np.uint8)

    mask_erode = cv2.erode(frame[:,:,2], kernel, iterations = 7)
    mask_dilate = cv2.dilate(mask_erode, kernel, iterations = 8)

	# Find the largest contour (sidewalk) in the mask
    (contours,_) = cv2.findContours(mask_dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if (len(contours) > 0):
        c_max = max(contours, key = cv2.contourArea)
        peri = cv2.arcLength(c_max, True)
        approx = cv2.approxPolyDP(c_max, 0.005 * peri, True)
        cv2.drawContours(frame, [approx], -1, (0, 0, 255), -1)

    cv2.imshow('Frame 2', frame)
    #cv2.imshow('Thresh R', thresh_r)
    #cv2.imshow('canny', canny)

    if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
        break

capture.release()
cv2.destroyAllWindows()