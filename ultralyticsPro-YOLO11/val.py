from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'runs/train/HIT-UAV/YOLOX/weights/last.pt')
    metrics = model.val(
        val=True,  
        data=r'data.yaml',
        split='val',  
        batch=16,  
        imgsz=640,  
        device='0',  
        workers=8,  
        save_json=False,  
        save_hybrid=False,  
        conf=0.001,  
        iou=0.6,  
        project='runs/val',  
        name='exp',  
        max_det=300,  
        half=False,  
        dnn=False,  
        plots=True,  
    )

    print(f"mAP50-95: {metrics.box.map}") 
    print(f"mAP50: {metrics.box.map50}")  
    print(f"mAP75: {metrics.box.map75}")  
    speed_metrics = metrics.speed
    total_time = sum(speed_metrics.values())
    fps = 1000 / total_time
    print(f"FPS: {fps}") 
    print(f"AP50: {metrics.box.ap50}")
    print(f"AP: {metrics.box.ap}")