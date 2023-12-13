import cv2
import os
from Trace import listdir_abs


def main():

    target = [0, 1, 2, 3, 4, 5, 6]
    video_path = r'E:\ICSHM2022_Database\data_project1~3\project1\video'
    output_dir = r'F:\ICSHM2022_output\tg_video'

    folders = listdir_abs(video_path)
    v_path = []
    o_path = []
    for tg in target:
        names = os.listdir(folders[tg])
        for name in names:
            v_path.append(folders[tg] + '\\' + name)
            o_path.append(output_dir + '\\' + name)
    for p, o in zip(v_path, o_path):
        video_to_frames(p, o, 5)


def video_to_frames(video_path, output_name, frame_frequency=1):
    # frame_frequency:提取视频的频率

    print('----Video capture target: ', output_name)
    times = 0

    # 如果文件目录不存在则创建目录
    if not os.path.exists(output_name):
        os.makedirs(output_name)

    # 读取视频帧
    camera = cv2.VideoCapture(video_path)

    while True:
        times = times + 1
        res, image = camera.read()
        if not res:
            break
        # 按照设置间隔存储视频帧
        if times % frame_frequency == 0:
            cv2.imwrite(output_name + '\\' + '%06d.jpg' % times, image)

    print('Video capture finished\n')
    # 释放摄像头设备
    camera.release()


if __name__ == '__main__':
    main()
