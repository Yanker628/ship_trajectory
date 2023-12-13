from ultralytics import YOLO

yaml_path = r'./dataset/ICSHM_BSCC.yaml'

# Load a model
model = YOLO('yolov8x.yaml').load('yolov8x.pt')  # build from YAML and transfer weights

div = [i for i in range(8)]

# Train the model
model.train(data=yaml_path, epochs=100, imgsz=1280, device=div,
            batch=24, optimizer='Adam', cos_lr=True, lr0=0.002)
