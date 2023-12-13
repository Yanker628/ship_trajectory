from ultralytics import YOLO
import cv2

model_path = r'E:\AICE\ICSHM2022\Pj_1\runs\detect\train2\weights\best.pt'
model = YOLO(model_path)
video = r'E:\ICSHM2022_Database\data_project1~3\project1\test2.mp4'
four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 设置输出视频为mp4格式
results = model.track(video, save=True, tracker="bytetrack.yaml")
