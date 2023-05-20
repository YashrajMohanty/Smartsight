import numpy as np
import cv2
from time import time, sleep
import torch
import math

class stereo_dist():

    def __init__(self):
        self.obstruction_flag = False

    def find_distance(self,disp, bb_center, display_text=True, fontscale=1):
        if bb_center == None:
            return
        if len(bb_center) == 0:
            return
        distances = []

        for i in bb_center:
            x, y = i
            x, y = int(x), int(y)

            distance = disp[y,x]
            distance = 1.42 + 40.08 * math.e**(-5.43 * distance)
            distance = round(distance, 1)
            distances.append(distance)

            if display_text == True:
                    cv2.putText(disp, str(distance)+'m', (x,y), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0,255,0), 1, cv2.LINE_AA)
        return distances

    def place_markers(self,frame):

        y_center, x_center = frame.shape[0]/2 , frame.shape[1]/2
        grid_size = 120
        grid = [int(-grid_size/2), 0, int(grid_size/2)]
        #grid = [0]

        marker_pos = []
        
        for i in grid:
            for j in grid:
                point = (x_center+i, y_center+j)
                marker_pos.append(point)

        dist = self.find_distance(frame, marker_pos, display_text=True, fontscale=0.5)

        near_count = 0
        for i in dist:
            if i < 3: # if distance returned by marker < 3.5 meters
                near_count += 1

        if near_count > 4: # if more than 4 markers return low distance
            self.obstruction_flag = True
        else:
            self.obstruction_flag = False
        return


class stereo_midas():

    def __init__(self):

        torch.hub.set_dir("Models/midas/")
        self.midas = torch.hub.load("Models/midas/intel-isl_MiDaS_master","MiDaS_small", source="local", verbose=False)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print('MiDaS using CUDA')
        else:
            self.device = torch.device("cpu")
            print('MiDaS using CPU')

        self.midas.to(self.device)
        self.midas.eval()
        midas_transforms = torch.hub.load("Models/midas/intel-isl_MiDaS_master","transforms", source="local")

        self.transform = midas_transforms.small_transform


    def predict_depth(self,frame):
        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (int(width/2), int(height/2)), interpolation=cv2.INTER_NEAREST) # half img size
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(frame).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
                ).squeeze()

        output = prediction.cpu().numpy()
        output = cv2.normalize(output,None,0,1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F) #64F prev
        output = cv2.resize(output, (width, height), interpolation=cv2.INTER_NEAREST) # half img size
        return output
    

    def convert_to_thermal(self, disp_map):
        thermal = disp_map * 255
        thermal = thermal.astype(np.uint8)
        thermal = cv2.applyColorMap(thermal, cv2.COLORMAP_MAGMA)
        return thermal
    
    
if __name__ == "__main__":

    midas = stereo_midas()
    sd = stereo_dist()

    prev_frame_time = 0
    new_frame_time = 0

    print('Engaging test')
    capture = cv2.VideoCapture('Chessboard/Stereo L anim.mp4')

    while capture.isOpened():

        _, frame = capture.read()

        if (str(type(frame))) == "<class 'NoneType'>":
            print('Stream ended')
            break

        disp_map = midas.predict_depth(frame)

        sd.place_markers(disp_map)

        new_frame_time = time()
        fps = int(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time
        cv2.putText(frame, str(fps), (0, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)

        cv2.imshow('frame',frame)

        
        cv2.imshow("Stereo", midas.convert_to_thermal(disp_map))
        
        #sleep(2)
        if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
            break

    capture.release()
    cv2.destroyAllWindows()