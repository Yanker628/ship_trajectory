from ultralytics import YOLO

yaml_path = r'E:\dataset\ICSHM_01.yaml'
model_path = r'E:\AICE\ICSHM2022\Pj_1\runs\detect\train6\weights\last.pt'

# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO(model_path)  # load a previous model (recommended for training)
model = YOLO('yolov8x.yaml').load('yolov8x.pt')  # build from YAML and transfer weights

# Train the model
model.train(data=yaml_path, epochs=25, imgsz=640, device=0, workers=0, batch=8, optimizer='Adam', cos_lr=False)
