import torch
from warnings import catch_warnings, simplefilter
with catch_warnings():
    simplefilter("ignore")
    from torchvision.transforms import v2
from custom_transforms import small_transform
import numpy as np
# 24mm horizontal=73.7deg vertical=41.45deg (iPhone 15 Pro Max main camera)


class measure_distance():

    def __init__(self):
        
        self.obstruction_flag = False
        self.caution_flag = False

    def __call__(self,frame):
        '''Measures values at certain points across the image

        Args:
            frame (tensor): image frame

        Returns:
            None
        '''

        y_center, x_center = int(frame.shape[0]/2) , int(frame.shape[1]/2)
        grid_size = 160
        grid_x = [-grid_size, int(-grid_size/2), 0, int(grid_size/2), grid_size]
        grid_y = [int(-grid_size/2), 0, int(grid_size/2)]

        dist = []
        
        for i in grid_x:
            for j in grid_y:
                dist.append(round(frame[y_center+j, x_center+i].item(), 2))
        
        #print(dist)
        #cv2.putText(frame, str(dist)+'m', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        near_count = 0
        far_count = 0
        for i in dist:
            if i > 0.75:
                near_count += 1
            if i < 0.2:
                far_count += 1

        self.obstruction_flag = False
        self.caution_flag = False

        if near_count >= 4: # if 4 or more markers return low distance
            self.obstruction_flag = True
                     
        if far_count == 15:
            self.caution_flag = True
            
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
        '''Predict image depth using MiDaS

        Args:
            frame (ndarray): image frame

        Returns:
            output (tensor): Depth map
        '''

        with torch.no_grad():
            frame = v2.ToImageTensor()(frame)
            frame = frame.to(self.device) #[channel, height, width]

            input_batch = small_transform(frame)

            prediction = self.midas(input_batch) #[1, 256, 256]

            output = torch.div(prediction, 1000) # map range from 0 to 1
            output = v2.Resize(size=(self.height, self.width), interpolation=v2.InterpolationMode.BICUBIC, antialias=False)(output)

        return output
    
    def top_view(self,img):
        horizontal_slice = img[350,:] * 480 # 350/480 row of image
        horizontal_slice = horizontal_slice.astype(np.int32)
        top_img = np.zeros((480,640))

        for i in range(len(horizontal_slice)):
            top_img[:horizontal_slice[i], i] = 1
        
        cv2.imshow('Top',top_img)
        return
    
    def side_view(self, img):
        vertical_range = img[:,290:350]
        vertical_max_slice = np.max(vertical_range, 1)
        vertical_max_slice = (vertical_max_slice * 640).astype(np.int32)
        side_img = np.zeros((480, 640))

        for i in range(len(vertical_max_slice)):
            side_img[i, :vertical_max_slice[i]] = 1

        cv2.imshow('Side', side_img)
            

    
    
if __name__ == "__main__":
    
    import cv2
    from time import time

    md = midas()
    measure_dist = measure_distance()

    prev_frame_time = 0
    new_frame_time = 0


    print('Engaging test')
    capture = cv2.VideoCapture('Chessboard/iphone 15 japan.mp4')

    while capture.isOpened():

        _, frame = capture.read()

        if type(frame) == type(None):
            print('Stream ended')
            break
        
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        disp_map = md.predict_depth(frame) # returns single channel image
        disp_map = disp_map.squeeze() # remove batch values of tensor
        disp_map_np = disp_map.cpu().numpy()

        md.side_view(disp_map_np)
        md.top_view(disp_map_np)

        new_frame_time = time()
        fps = int(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time


        cv2.putText(disp_map_np, str(fps), (0, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
        cv2.imshow('Disp',disp_map_np)
        
        #sleep(2)
        if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
            break

    capture.release()
    cv2.destroyAllWindows()