from ultralytics import YOLO
import gc
from Trace import Trace
import os

model_path = r'E:\AICE\ICSHM2022\Pj_1\runs\detect\train2\weights\last.pt'
model = YOLO(model_path)
trace = Trace(model=model)

video_dir = r'F:\ICSHM2022_output\tg_video'

for video in os.listdir(video_dir)[1:]:
    video_path = os.path.join(video_dir, video)
    trace.predict_video(video_path, stream=False, wait_key=10, confidence=0.1, reverse=False, save_video=True,
                        verbose=False)
    gc.collect()

# video_dir = r'F:\ICSHM2022_output\video\test2.mp4'
trace.predict_video(video_dir, stream=True, wait_key=10, confidence=0.2, reverse=False,
                    save_video=True, verbose=False)
