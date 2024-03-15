# Smartsight

## Overview
A python based application that aids visually impaired individuals with basic navigation with the help of image processing and deep learning techniques. The program works by recording frames from a camera (or video) and running inference on the MiDaS model followed by other forms of processing.


## The program consists of the following modules:   


### Depth_DNN.py
This runs the main loop of the program. It generates depth maps and further processes it to obtain appropriate data from the input images. It utilizes the Pytorch [MiDaS](https://pytorch.org/hub/intelisl_midas_v2/) model to perform depth estimation.


![MiDaS Depth Map](overview_img/midas_img.png)


### Audio_feedback.py
The data provided by Depth_DNN.py is used to generate auditory feedback.


## Usage/Inference
Set the necessary paths and **Depth_DNN.py** and run it.


To use the camera, set the source in **Depth_DNN.py** to the device index:  
`capture = cv2.VideoCapture(0)`  
Or add the path to an **mp4** file:  
`capture = cv2.VideoCapture("Test_video.mp4")`  

While the program is running, use `q` to quit.
