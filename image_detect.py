from ultralytics import YOLO
import gc
from Trace import Trace
import os
import cv2

model_path = r'E:\AICE\ICSHM2022\Pj_1\runs\detect\train2\weights\best.pt'
model = YOLO(model_path)
trace = Trace(model=model)
pic = r'F:\ICSHM2022_output\tg_video\3-3.mp4\002000.jpg'

result = model.predict(pic, nms_disable=True)
cv2.imshow('res', result[0].plot())
cv2.waitKey(0)
