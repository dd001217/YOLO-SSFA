import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO



if __name__ == '__main__':
    model = YOLO('runs/train/FLIR/YOLO11/weights/best.pt') 
    model.predict(source='datasets/FLIR/test/images/FLIR_video_03341.jpeg',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                )