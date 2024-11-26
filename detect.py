from ultralytics.models import YOLO
 
 
if __name__ == '__main__':
    model = YOLO(model='yolov11/yoloV11-xiaoxin/runs/train/exp9/weights/best.pt')
    model.predict(source='yolov11/yoloV11-xiaoxin/data/images', device='0', imgsz=640, save=True, conf=0.2, iou=0.7, project='yolov11/yoloV11-xiaoxin/runs/detect/', name='exp')