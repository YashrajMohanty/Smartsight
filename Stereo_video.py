import numpy as np
import cv2
from glob import glob
import math

img_shape = [640, 480]
disp = None

class calibrate_cam:

    objpoints = None
    imgpoints = None
    mtx = None
    dist = None

    def __init__(self, folder):
        self.folder = folder


    def calibrate_camera(self):
        folder = self.folder
            # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        for filename in list(glob(folder+'/*.jpg')):
            img = cv2.imread(filename)
            #img = cv2.resize(img, (0, 0), fx = 0.625, fy = 0.625)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
        #new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img.shape[:2], 1, img.shape[:2])
        
        print("Camera matrix:\n", mtx)
        print("\nDistortion coefficient:\n", dist)
        print("\nRotation Vectors:\n", rvecs)
        print("\nTranslation Vectors:\n", tvecs)
        print('--------------------------------------')

        self.objpoints = objpoints
        self.imgpoints = imgpoints
        self.mtx = mtx
        self.dist = dist

class stereo_cam:
    disp = None
    Left_Stereo_Map = None
    Right_Stereo_Map = None
    stereo = None
    stereoR = None
    wls_filter = None
    min_disp = None
    num_disp = None
    Q = None

    def find_distance(frame, bb_center, display_text=True, fontscale=1):
        if bb_center == None:
            return
        if len(bb_center) == 0:
            return
        distances = []
        for i in bb_center:
            x, y = i
            disp = stereo_cam.disp
            average=0
            for u in range (-1,2):
                for v in range (-1,2):
                    average += disp[y+u,x+v]
            average=average/9
            Distance= (28.6275*(average**2)) - (39.1097*average) + 14.9541
            #Distance= np.around(Distance*0.01,decimals=2)
            Distance = 2.4*Distance
            Distance = round(Distance, 2)
            distances.append(Distance)

            if display_text == True:
                if Distance < 10:
                    cv2.putText(frame, str(Distance)+'m', (x,y), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0,0,255), 1, cv2.LINE_AA)
                else:
                    cv2.putText(frame, 'INF', (x,y), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0,0,255), 1, cv2.LINE_AA)
        return distances

    def calibrate_stereo():

        camL = calibrate_cam('Chessboard/L')
        camL.calibrate_camera()
        camR = calibrate_cam('Chessboard/R')
        camR.calibrate_camera()

        criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        _, MLS, dLS, MRS, dRS, R, T, _, _ = cv2.stereoCalibrate(camL.objpoints, #objpoints same for both cameras
                                                                camL.imgpoints,
                                                                camR.imgpoints,
                                                                camL.mtx,
                                                                camL.dist,
                                                                camR.mtx,
                                                                camR.dist,
                                                                img_shape,
                                                                criteria = criteria_stereo,
                                                                flags = cv2.CALIB_FIX_INTRINSIC)

        RL, RR, PL, PR, Q, _, _= cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                        img_shape, R, T,
                                                        1,(0,0))  # last paramater is alpha, if 0= cropped, if 1= not cropped

        Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                                    img_shape, cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the program to work faster
        Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                                    img_shape, cv2.CV_16SC2)

        # Create StereoBM and prepare all parameters  
        min_disp = 2
        num_disp = 32#113-min_disp
        stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=5)

        # Used for the filtered image
        stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right camera

        # WLS FILTER Parameters
        lmbda = 80000
        sigma = 1.8
        
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)

        stereo_cam.Left_Stereo_Map = Left_Stereo_Map
        stereo_cam.Right_Stereo_Map = Right_Stereo_Map
        stereo_cam.stereo = stereo
        stereo_cam.stereoR = stereoR
        stereo_cam.wls_filter = wls_filter
        stereo_cam.min_disp = min_disp
        stereo_cam.num_disp = num_disp
        stereo_cam.Q = Q

    def place_squares(img):

        x_center = 320
        y_center = 240
        square_size = 50

        c1 = (x_center-square_size,y_center-square_size)
        c2 = (x_center,y_center-square_size)
        c3 = (x_center+square_size,y_center-square_size)
        c4 = (x_center-square_size,y_center)
        c5 = (x_center,y_center)
        c6 = (x_center+square_size,y_center)
        c7 = (x_center-square_size,y_center+square_size)
        c8 = (x_center,y_center+square_size)
        c9 = (x_center+square_size,y_center+square_size)

        square_pos = [c1,c2,c3,c4,c5,c6,c7,c8,c9]
        stereo_cam.find_distance(img, square_pos, display_text=True, fontscale=0.5)
        return

    def run_stereo(frameL, frameR):

        Left_Stereo_Map = stereo_cam.Left_Stereo_Map
        Right_Stereo_Map = stereo_cam.Right_Stereo_Map
        stereo = stereo_cam.stereo
        num_disp = stereo_cam.num_disp
        wls_filter = stereo_cam.wls_filter

        Left_nice= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)  # Rectify the image using the calibration parameters founds during the initialisation
        Right_nice= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)
    
        grayR= cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
        grayL= cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)
 
        disp = stereo.compute(grayL,grayR).astype(np.float32) / 16.0
        disp = disp / num_disp
        disp = wls_filter.filter(disp,grayL,None,disp)

        #baseline = 0.16 # distance between cameras in cm
        #foc_len = 0.03 # focal length of camera in cm
        #depth = (baseline * foc_len) / disp
        #depth_thresh = 5
        #mask = cv2.inRange(depth, 4, depth_thresh)

        stereo_cam.disp = disp
        return disp


if __name__ == "__main__":
    
    stereo_cam.calibrate_stereo()
    print('Engaging test')
    captureL = cv2.VideoCapture('Chessboard/Stereo L anim.mp4')
    captureR = cv2.VideoCapture('Chessboard/Stereo R anim.mp4')

    while captureL.isOpened():

        _, frameL = captureL.read()
        _, frameR = captureR.read()

        if (str(type(frameL))) == "<class 'NoneType'>":
            print('Stream ended')
            break

        #disp_map = stereo_cam.run_stereo(frameL, frameR)
        disp_map = stereo_cam.run_stereo(frameL, frameR)
        stereo_cam.place_squares(disp_map)
        cv2.imshow("Stereo", disp_map)
        
        if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
            break

    captureL.release()
    captureR.release()
    cv2.destroyAllWindows()