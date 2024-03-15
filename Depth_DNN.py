import torch
from warnings import catch_warnings, simplefilter
with catch_warnings():
    simplefilter("ignore")
    from torchvision.transforms import v2
import numpy as np
import Audio_feedback
# 24mm horizontal=73.7deg vertical=41.45deg (iPhone 15 Pro Max main camera)
# y = (-0.11)x**2 + 1.07 (Parabolic eq)


class midas():

    def __init__(self):

        torch.hub.set_dir("Models/midas/")
        self.midas = torch.hub.load("Models/midas/intel-isl_MiDaS_master","MiDaS_small", source="local", verbose=False)

        self.midas_input_transform = v2.Compose(
        [
            lambda img: {"image": img / 255.0},
            v2.Resize([256,256], interpolation=v2.InterpolationMode.BICUBIC, antialias=False),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            lambda sample: (sample["image"]).unsqueeze(0),
        ])

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

            input_batch = self.midas_input_transform(frame)

            prediction = self.midas(input_batch) #[1, 256, 256]

            output = v2.Resize(size=(96, 128), interpolation=v2.InterpolationMode.BICUBIC, antialias=False)(prediction)
            output = torch.div(output, 1000) # map range from 0 to 1
            output = torch.where(output > 1, 1, output)

            output = output.squeeze() # remove batch values of tensor
            output = output.cpu().numpy() # move tensor to CPU and convert to numpy
        return output


def top_view(img):
    top_slice = img[75,:] * 100 # 75/96 row of image
    top_slice = top_slice.astype(np.int32)    
    return top_slice


def draw_top_view(slice):
    slice = (slice * 0.96).astype(np.int32)
    top_img = np.zeros((96,128))
    for i in range(len(slice)):
        top_img[:slice[i], i] = 1
    cv2.imshow('Top',top_img)


def top_spatial_data(top_slice):
    near_slice = top_slice == 100
    near_left = near_slice[:64].sum()
    near_right = near_slice[64:].sum()
    #net_y_motion = near_right - near_left
    #total_y_motion = near_right + near_left

    return near_left, near_right


def motion(slice1, slice0):
    x_diff = slice1 - slice0
    x_motion = -0.11 * x_diff * x_diff + 1.07 # y = (-0.11)x**2 + 1.07 (Parabolic eq)
    x_motion = x_motion.astype(np.int32)
    return

#def side_view(img):
#    vertical_range = img[:,110:146]
#    vertical_max_slice = np.max(vertical_range, 1)
#    vertical_max_slice = (vertical_max_slice * 256).astype(np.int32)
#    return vertical_max_slice

#def draw_side_view(slice):
#    side_img = np.zeros((192,256))
#    for i in range(len(slice)):
#        side_img[i, :slice[i]] = 1
#    cv2.imshow('Side', side_img)
             
    
if __name__ == "__main__":
    
    import cv2

    md = midas()
    Audio_feedback.audio_thread.start()

    slice0 = np.zeros((1,128),dtype=np.int32)

    print('Engaging test')
    capture = cv2.VideoCapture('Test_videos/iphone 15 japan.mp4')

    while capture.isOpened():

        _, frame = capture.read()

        if type(frame) == type(None):
            print('Stream ended')
            break
        
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        disp_map = md.predict_depth(frame) # returns single channel image
        top_slice = top_view(disp_map)
        #draw_top_view(top_slice)
        near_left, near_right = top_spatial_data(top_slice)
        Audio_feedback.set_freq(near_left, near_right)
        #motion(slice0,top_slice)
        #slice0 = top_slice


        cv2.imshow('Disp',disp_map)
        
        #sleep(2)
        if cv2.waitKey(10) & 0xFF == ord('q'): #press q to quit
            break

    capture.release()
    cv2.destroyAllWindows()