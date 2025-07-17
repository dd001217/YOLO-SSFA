import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v10/yolov10n.yaml')
    model.train(data='D:/Desktop/ultralyticsPro-YOLO11/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=10,
                workers=2,
                optimizer='SGD',
                project='runs/train',
                name='exp',
                )