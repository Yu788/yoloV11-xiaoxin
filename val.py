from ultralytics.models import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
 
if __name__ == '__main__':
    model = YOLO(model='yolov11/yoloV11-xiaoxin/runs/train/exp9/weights/best.pt')
    model.val(data='data.yaml', split='val', batch=4, device='0', imgsz=640, project='yolov11/yoloV11-xiaoxin/runs/val', name='exp',
              half=False,)