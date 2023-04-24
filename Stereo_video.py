import numpy as np
import cv2
from glob import glob

img_shape = [640, 480] # original img size
img_shape_half = [320, 240]

class calibrate_cam:

    def __init__(self, folder):
        self.folder = folder
        self.objpoints = None
        self.imgpoints = None
        self.mtx = None
        self.dist = None


    def calibrate_camera(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        for filename in list(glob(self.folder+'/*.jpg')):

            img = cv2.imread(filename)
            img = cv2.resize(img, img_shape_half) # half the img size
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape_half, None, None)
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

    def __init__(self):
        self.disp = None
        self.Left_Stereo_Map = None
        self.Right_Stereo_Map = None
        self.stereo = None
        self.wls_filter = None
        self.num_disp = None
        self.obstruction_flag = False

    def find_distance(self, frame, bb_center, display_text=True, fontscale=1):
        if bb_center == None:
            return
        if len(bb_center) == 0:
            return
        distances = []
        disp = self.disp
        for i in bb_center:
            x, y = i
            x, y = int(x), int(y)
            avg = disp[y,x]
            if avg == 0:
                avg = 0.01
            avg = abs(avg)
            if avg < 0.217:
                distance = 10
            elif avg > 0.72:
                distance = 1
            else:
                distance = (5.3043/(avg**0.4042)) - 5.2340
            distance = round(distance, 2)
            distances.append(distance)

            if display_text == True:
                if distance < 8 and distance > 3.5:
                    cv2.putText(frame, str(distance)+'m', (x,y), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0,255,0), 1, cv2.LINE_AA)
                else:
                    cv2.putText(frame, str(distance)+'m', (x,y), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0,0,255), 1, cv2.LINE_AA)
        return distances

    def calibrate_stereo(self):

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
                                                                img_shape_half,
                                                                criteria = criteria_stereo,
                                                                flags = cv2.CALIB_FIX_INTRINSIC)

        RL, RR, PL, PR, _, _, _= cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                        img_shape_half, R, T,
                                                        1,(0,0))

        Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                                    img_shape_half, cv2.CV_16SC2)  # cv2.CV_16SC2 allows program to run faster
        Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                                    img_shape_half, cv2.CV_16SC2)

        # Create StereoBM and prepare all parameters  
        num_disp = 64
        stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=5)

        # WLS FILTER Parameters 
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
        wls_filter.setLambda(80000) #lambda
        wls_filter.setSigmaColor(1.8) #sigma

        self.Left_Stereo_Map = Left_Stereo_Map
        self.Right_Stereo_Map = Right_Stereo_Map
        self.stereo = stereo
        self.wls_filter = wls_filter
        self.num_disp = num_disp

    def place_markers(self, img):

        x_center = (img_shape[0] / 2)  # shifting markers right by 50 px
        y_center = img_shape[1] / 2
        grid_size = 180
        grid = [int(-grid_size/2), 0, int(grid_size/2)]

        marker_pos = []
        
        for i in grid:
            for j in grid:
                point = (x_center+i, y_center+j)
                marker_pos.append(point)

        near_count = 0
        dist = self.find_distance(img, marker_pos, display_text=True, fontscale=0.5)

        for i in dist:
            if i < 3: # if distance returned by marker < 3
                near_count += 1

        if near_count > 3: # if more than 3 markers return low distance
            self.obstruction_flag = True
        else:
            self.obstruction_flag = False
        return

    def tranform_input_img(self, frame):
        frame = cv2.resize(frame, img_shape_half) # half img size
        frame = cv2.medianBlur(frame,3) # apply median blur
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # convert from BGR to HSV
        _,_,v = cv2.split(frame) # split channels
        return v

    def run_stereo(self, frameL, frameR):

        Left_Stereo_Map = self.Left_Stereo_Map
        Right_Stereo_Map = self.Right_Stereo_Map
        stereo = self.stereo
        num_disp = self.num_disp
        wls_filter = self.wls_filter

        frameL = self.tranform_input_img(frameL)
        frameR = self.tranform_input_img(frameR)

        Left_remap = cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)  # Rectify the image using the calibration parameters founds during the initialisation
        Right_remap = cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], interpolation = cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)

        grayL= Left_remap # cv2.cvtColor(Left_remap,cv2.COLOR_BGR2GRAY)
        grayR= Right_remap
 
        disp = stereo.compute(grayL,grayR).astype(np.float32) / 16
        disp = disp / num_disp
        #disp = cv2.normalize(disp,(0,0),0,2,cv2.NORM_MINMAX)
        disp = wls_filter.filter(disp,grayL,None,disp)
        disp = 1 - disp[:,64:] # inverting and cropping image
        disp = cv2.resize(disp, img_shape) # resize img to full size
        self.disp = disp
        return disp


if __name__ == "__main__":
    
    stereo = stereo_cam()
    stereo.calibrate_stereo()

    print('Engaging test')
    captureL = cv2.VideoCapture('Chessboard/Stereo L anim.mp4')
    captureR = cv2.VideoCapture('Chessboard/Stereo R anim.mp4')

    while captureL.isOpened():

        _, frameL = captureL.read()
        _, frameR = captureR.read()

        if (str(type(frameL))) == "<class 'NoneType'>":
            print('Stream ended')
            break

        disp_map = stereo.run_stereo(frameL, frameR)
        stereo.place_markers(disp_map)
        cv2.imshow("Stereo", disp_map)
        
        if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
            break

    captureL.release()
    captureR.release()
    cv2.destroyAllWindows()