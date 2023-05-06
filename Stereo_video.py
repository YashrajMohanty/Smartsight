import numpy as np
import cv2
from glob import glob
import os

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

        print('Calibrating camera')
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.

        for filename in list(glob(self.folder+'/*.jpg')):

            img = cv2.imread(filename)
            img = cv2.resize(img, img_shape_half) # half the img size
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                self.imgpoints.append(corners2)

        _, self.mtx, self.dist, _, _ = cv2.calibrateCamera(self.objpoints, self.imgpoints, img_shape_half, None, None)
        #new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, img.shape[:2], 1, img.shape[:2])
        
        print("Camera matrix:\n", self.mtx)
        print("\nDistortion coefficient:\n", self.dist)
        print('--------------------------------------')


class stereo_cam:

    def __init__(self):
        self.disp = None

        self.Left_Stereo_Map = None
        self.Right_Stereo_Map = None

        # prepare stereoBM with parameters
        self.num_disp = 64
        self.stereoL = cv2.StereoBM_create(numDisparities=self.num_disp, blockSize=5)
        self.stereoL.setTextureThreshold(10)
        self.stereoL.setDisp12MaxDiff(10)

        # WLS FILTER Parameters
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.stereoL)
        self.wls_filter.setLambda(80000) #lambda
        self.wls_filter.setSigmaColor(1.8) #sigma

        self.obstruction_flag = False


    def find_distance(self, frame, bb_center, display_text=True, fontscale=1):
        if bb_center == None:
            return
        if len(bb_center) == 0:
            return
        distances = []

        for i in bb_center:
            x, y = i
            x, y = int(x), int(y)

            if x > 128:    
                distance = self.disp[y,x]
                if distance > 0.55:
                    distance = 10
                elif distance < 0.1:
                    distance = 1
                else:
                    distance = (57.02 * (distance**3.12)) + 0.88
                    distance = distance * 1.8
                    distance = round(distance, 2)
            else:
                distance = 20

            distances.append(distance)

            if display_text == True:
                if distance < 8 and distance > 3.5:
                    cv2.putText(frame, str(distance)+'m', (x,y), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0,255,0), 1, cv2.LINE_AA)
                else:
                    cv2.putText(frame, str(distance)+'m', (x,y), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0,0,255), 1, cv2.LINE_AA)
        return distances


    def load_params(self):
        print('Loading saved parameters')
        Left_param1 = np.load('Chessboard/L/calibData/Left_Stereo_Map1.npy')
        Left_param2 = np.load('Chessboard/L/calibData/Left_Stereo_Map2.npy')
        Right_param1 = np.load('Chessboard/R/calibData/Right_Stereo_Map1.npy')
        Right_param2 = np.load('Chessboard/R/calibData/Right_Stereo_Map2.npy')

        Left_stereo_map = [Left_param1, Left_param2]
        Right_stereo_map = [Right_param1, Right_param2]
        
        print('Parameters loaded')
        return (Left_stereo_map, Right_stereo_map)


    def save_params(self, Left_param, Right_param):
        print('Saving parameters')
        Left_param1 = Left_param[0]
        Left_param2 = Left_param[1]
        Right_param1 = Right_param[0]
        Right_param2 = Right_param[1]

        save_directoryL = 'Chessboard/L/calibData/'
        save_directoryR = 'Chessboard/R/calibData/'

        np.save(save_directoryL+'Left_Stereo_Map1.npy', Left_param1, allow_pickle=False, fix_imports=False)
        np.save(save_directoryL+'Left_Stereo_Map2.npy', Left_param2, allow_pickle=False, fix_imports=False)
        np.save(save_directoryR+'Right_Stereo_Map1.npy', Right_param1, allow_pickle=False, fix_imports=False)
        np.save(save_directoryR+'Right_Stereo_Map2.npy', Right_param2, allow_pickle=False, fix_imports=False)
        print('Save complete')
        return


    def calibrate_stereo(self, force_calibrate=False):

        if (len(os.listdir('Chessboard/L/calibData')) == 2) and (len(os.listdir('Chessboard/R/calibData')) == 2) and not force_calibrate:
            self.Left_Stereo_Map, self.Right_Stereo_Map = self.load_params()

        else:
            print('Beginning calibration')
            camL = calibrate_cam('Chessboard/L')
            camR = calibrate_cam('Chessboard/R')
            camL.calibrate_camera()
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

            self.Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                                        img_shape_half, cv2.CV_16SC2)
            self.Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                                        img_shape_half, cv2.CV_16SC2)

            self.save_params(self.Left_Stereo_Map, self.Right_Stereo_Map)


    def run_stereo(self, frameL, frameR):

        frameL = self.tranform_input_img(frameL)
        frameR = self.tranform_input_img(frameR)

        Left_remap = cv2.remap(frameL, self.Left_Stereo_Map[0], self.Left_Stereo_Map[1], interpolation=cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)  # Rectify the image using the calibration parameters found during the initialisation
        Right_remap = cv2.remap(frameR, self.Right_Stereo_Map[0], self.Right_Stereo_Map[1], interpolation=cv2.INTER_LANCZOS4, borderMode = cv2.BORDER_CONSTANT)

        grayL= Left_remap # cv2.cvtColor(Left_remap,cv2.COLOR_BGR2GRAY)
        grayR= Right_remap # cv2.cvtColor(Left_remap,cv2.COLOR_BGR2GRAY)

        dispL = self.stereoL.compute(grayL,grayR).astype(np.float32) / 16
        dispL = dispL / self.num_disp
        dispL = dispL + 0.015625 # cropping and correcting image

        disp = self.wls_filter.filter(dispL,grayL,None,dispL)
        #disp = self.fix_light(disp)
        disp = disp.clip(min=0, max=1)
        #disp = disp[:,64:].clip(min=0)
        #disp = cv2.normalize(disp,(0,0),0,2,cv2.NORM_MINMAX)
        disp = cv2.resize(disp, (disp.shape[1]*2, disp.shape[0]*2)) # resize img to full size
        #print(disp.max(),disp.min())
        
        #disp = cv2.resize(disp, img_shape) # resize img to full size
        self.disp = disp
        return disp
    

    def tranform_input_img(self, frame):
        frame = cv2.resize(frame, img_shape_half) # half img size
        #frame = cv2.medianBlur(frame,3) # apply median blur
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # convert from BGR to HSV
        _,_,frame = cv2.split(frame) # split channels
        return frame
    

    def place_markers(self, img):

        x_center = img_shape[0] / 2  # shifting markers right by 50 px
        y_center = img_shape[1] / 2
        grid_size = 120
        grid = [int(-grid_size/2), 0, int(grid_size/2)]

        marker_pos = []
        
        for i in grid:
            for j in grid:
                point = (x_center+i, y_center+j)
                marker_pos.append(point)

        near_count = 0
        dist = self.find_distance(img, marker_pos, display_text=True, fontscale=0.5)

        for i in dist:
            if i < 3: # if distance returned by marker < 3.5 meters
                near_count += 1

        if near_count > 4: # if more than 4 markers return low distance
            self.obstruction_flag = True
        else:
            self.obstruction_flag = False
        return
    

    def fix_light(self, ndarray):
        ref70percentile = 0.71
        img70percentile = np.percentile(ndarray, 70)
        factor = ref70percentile / img70percentile
        img = ndarray * factor
        return img


if __name__ == "__main__":
    
    stereo = stereo_cam()
    stereo.calibrate_stereo(force_calibrate=False)

    print('Engaging test')
    captureL = cv2.VideoCapture('Chessboard/Stereo L anim.mp4')
    captureR = cv2.VideoCapture('Chessboard/Stereo R anim.mp4')

    while captureL.isOpened():

        _, frameL = captureL.read()
        _, frameR = captureR.read()

        if (str(type(frameL))) == "<class 'NoneType'>":
            print('Stream ended')
            break
        
        cv2.imshow('frame',frameL)
        #cv2.imshow('R', frameR)
        disp_map = stereo.run_stereo(frameL, frameR)
        stereo.place_markers(disp_map)
        cv2.imshow("Stereo", disp_map)
        
        if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
            break

    captureL.release()
    captureR.release()
    cv2.destroyAllWindows()