from ultralytics.models import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
 
if __name__ == '__main__':
    model = YOLO(model='yolov11/yoloV11-xiaoxin/ultralytics/cfg/models/11/yolo11.yaml')
    model.load('yolo11n.pt')
    model.train(data='yolov11/yoloV11-xiaoxin/data.yaml', epochs=200, batch=4, device='0', imgsz=640, workers=0, cache=False,
                amp=True, mosaic=False, project='yolov11/yoloV11-xiaoxin/runs/train', name='exp',optimizer='SGD')