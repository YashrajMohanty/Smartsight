import numpy as np
import cv2
from time import time
import torch
from math import e
from torchvision.transforms import v2
from custom_transforms import small_transform


class distance_estimation():

    def __init__(self):
        self.obstruction_flag = False
        self.caution_flag = False

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
            distance = 1.42 + 40.08 * e**(-5.43 * distance)
            distance = round(distance, 1)
            distances.append(distance)

            if display_text == True:
                    cv2.putText(disp, str(distance)+'m', (x,y), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0,255,0), 1, cv2.LINE_AA)
        return distances

    def place_markers(self,frame):

        y_center, x_center = frame.shape[0]/2 , frame.shape[1]/2
        grid_size = 160
        grid_x = [-grid_size, int(-grid_size/2), 0, int(grid_size/2), grid_size]
        grid_y = [int(-grid_size/2), 0, int(grid_size/2)]
        #grid = [0]

        marker_pos = []
        
        for i in grid_x:
            for j in grid_y:
                point = (x_center+i, y_center+j)
                marker_pos.append(point)

        dist = self.find_distance(frame, marker_pos, display_text=True, fontscale=0.5)

        near_count = 0
        far_count = 0
        for i in dist:
            if i < 3: # if distance returned by marker < 3 meters
                near_count += 1
            if i > 15: # if distance returned by marker > 15 meters
                far_count += 1

        if near_count >= 4: # if 4 or more markers return low distance
            self.obstruction_flag = True
        else:
            self.obstruction_flag = False

        if far_count == 15:
            self.caution_flag = True
        else:
            self.caution_flag = False
        return


class midas():

    def __init__(self):

        self.height = 480
        self.width = 640

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

        #self.midas_transform = torch.hub.load("Models/midas/intel-isl_MiDaS_master","transforms", source="local").small_transform



    def predict_depth(self,frame):

        input_batch = small_transform(frame)
        print(input_batch.shape)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(0), #.unsqueeze()
                size=[256,256],#frame.shape[:2],
                mode="bicubic",
                align_corners=False).squeeze(1)

        print(prediction.dtype, prediction.shape)
        output = v2.Resize(size=(self.height, self.width), interpolation=v2.InterpolationMode.BICUBIC, antialias=False)(prediction)
        output = torch.div(output, 1000) # map range from 0 to 1
        #output = cv2.resize(output, (self.width, self.height), interpolation=cv2.INTER_NEAREST) # half img size

        return output
    

    def convert_to_thermal(self, disp_map):
        thermal = disp_map * 255
        thermal = thermal.astype(np.uint8)
        thermal = cv2.applyColorMap(thermal, cv2.COLORMAP_MAGMA)
        return thermal
    
    
if __name__ == "__main__":

    midas = midas()
    sd = distance_estimation()

    prev_frame_time = 0
    new_frame_time = 0

    print('Engaging test')
    capture = cv2.VideoCapture('Chessboard/Stereo L anim.mp4')

    while capture.isOpened():

        _, frame = capture.read()

        if (str(type(frame))) == "<class 'NoneType'>":
            print('Stream ended')
            break

        frame = torch.from_numpy(frame)
        frame = frame.to(midas.device)
        frame = frame.permute([2,0,1])
        #print(frame.shape) #[channel, height, width]

        disp_map = midas.predict_depth(frame) # returns single channel image
        disp_map = disp_map.squeeze() # remove batch values of tensor
        disp_map = disp_map.cpu().numpy()

        frame = frame.permute([1,2,0])
        frame = frame.cpu().numpy()
        #sd.place_markers(disp_map)

        new_frame_time = time()
        fps = int(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time
        cv2.putText(disp_map, str(fps), (0, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)

        #cv2.imshow('frame',frame)
        cv2.imshow('Disp',disp_map)

        
        #cv2.imshow("Stereo", midas.convert_to_thermal(disp_map))
        
        #sleep(2)
        if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
            break

    capture.release()
    cv2.destroyAllWindows()