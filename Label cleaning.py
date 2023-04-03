from glob import glob
from ultralytics import YOLO
from os.path import isfile

def clean_labels():
    for filename in list(glob('coco128/train/labels/*.txt')):
        with open(filename, "r") as txt_read:
            lines = txt_read.readlines()
        txt_read.close()
        with open(filename, "w") as txt_write:
            for line in lines:
                if len(lines) == 0:
                    txt_write.write('')
                    break
                if line[:2]!='-1':
                    txt_write.write(line)

            txt_write.flush()
        txt_write.close()
    print('Success')
    return

def train_model():
    model = YOLO('Custommodel.yaml').load('ultralytics/yolov8n.pt')  # build from YAML and transfer weights
    model.train(data='Custommodel.yaml', epochs=100, imgsz=640, task='detect')
    return

def del_label():
    lost = []
    for filename in list(glob('coco128/train/images/*.jpg')):
        filename = filename.split('/')[2]
        filename = filename[7:-4]
        if not isfile('coco128/train/newlabels/'+filename+'.txt'):
            lost.append(filename)
    print(lost)
    print(len(lost))
    
clean_labels()