# Smartsight
A python based application that aids visually impaired individuals with basic navigation with the help of computer vision technology. It utilizes object detection
along with other image processing techniques to provide helpful information about the environment.

The program works by recording video (frames) in two cameras (left and right) and passing the frames to different modules to obtain the required results.

Object detection uses the YOLOv8 model to detect object classes and provide bounding boxes.

The stereo vision module uses the left and right frames to generate disparity maps and calculate distances. 

The information provided by these modules are used by the audio feedback module to generate and play the suitable alerts.
