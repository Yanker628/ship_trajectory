import os.path
import time

import cv2
import xlrd
import warnings
import numpy as np
import os
from datetime import datetime, timedelta
import json
from time import mktime
import gc
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from Trace import Boat

COLOR_MAP = ['red', 'orange', 'yellow', 'lime', 'aqua', 'blue', 'fuchsia',
             'darkred', 'peru', 'darkgoldenrod', 'darkgreen', 'teal', 'darkblue', 'purple',
             'coral', 'khaki', 'mediumaquamarine', 'dodgerblue', 'mediumslateblue', 'deeppink']


class Time:
    """
    输入的t_number格式：年月日时分秒+4位小数,共18位,数据格式为整数
    年月日默认为2022.1.1，时分秒默认0
    可用对象:
       a_time：暂存
       time：datetime.strptime
       n_time：原始输入，年月日时分秒+4位小数,共18位,数据格式为整数
       str_time：输出可读的时间，用于展示
       is_later, not_earlier：判断是否比输入时间晚，返回T/F，后者可以取等
       add：加法运算，x为加上的秒数，以Time返回相加后得到的时间
       subtract：减法运算，x不能含有年或月，以秒数返回相减后得到的时间
       seconds：以float返回这个时间（月份以下）有多少秒，用于计算，
    """

    def __init__(self, t_number: int = 0):
        chara = str('%018d' % np.absolute(t_number))
        # 默认值
        if t_number < 0:
            chara = '202201010000000000'
        if len(chara) > 18:
            warnings.warn('Time input longer than 18')
            chara = chara[:18]
        self.n_time = int(chara)
        self.__at()
        self.time = datetime.strptime(self.str_time(), "%Y/%m/%d %H:%M:%S:%f")

    def __at(self):
        # 刷新a_time
        self.a_time = [0, 0, 0, 0, 0, 0, 0]
        j, k = 0, 0
        for i in [4, 2, 2, 2, 2, 2, 4]:
            self.a_time[k] = int(str('%018d' % self.n_time)[j: j + i])
            k += 1
            j += i
        j = 0
        for i in [9999, 12, 31, 23, 59, 59]:
            if int(self.a_time[j]) > i:
                if j < 3:
                    self.a_time[j] = self.a_time[j] % i
                else:
                    self.a_time[j] = self.a_time[j] % (i + 1)
                    self.a_time[j + 1] = self.a_time[j + 1] + 1

            j += 1

    def str_time(self, interval: str = '/') -> str:
        i = interval
        return '%04d%s%02d%s%02d %02d:%02d:%02d:%04d' % \
            (self.a_time[0], i, self.a_time[1], i, self.a_time[2], self.a_time[3], self.a_time[4], self.a_time[5],
             self.a_time[6])

    def is_later(self, x) -> bool:
        if self.time > x.time:
            return True
        else:
            return False

    def not_earlier(self, x) -> bool:
        if self.time >= x.time:
            return True
        else:
            return False

    def add(self, x: float):
        new_time = self.time + timedelta(seconds=x)
        return self.time_f_datetime(new_time)

    def subtract(self, x):
        return (self.time - x.time).seconds

    def seconds(self) -> float:
        base_time = Time(-1)
        return (self.time - base_time.time).seconds

    @staticmethod
    def time_f_num(x):
        ms = x % 10000
        ss = x // 10000
        s = ss % 60
        mm = ss // 60
        m = mm % 60
        hh = mm // 60
        h = hh % 24
        d = hh // 24
        return Time(int(str('%02d%02d%02d%02d%04d' % (d, h, m, s, ms))))

    @staticmethod
    def time_f_datetime(t):
        t_num = int(t.year)
        t_num = t_num * 100 + int(t.month)
        t_num = t_num * 100 + int(t.day)
        t_num = t_num * 100 + int(t.hour)
        t_num = t_num * 100 + int(t.minute)
        t_num = t_num * 100 + int(t.second)
        t_num = t_num * 10000 + int(t.microsecond // 100)
        return Time(t_num)

    def __repr__(self):
        return self.str_time()

    def __str__(self):
        return self.str_time()


class MAP:
    CLASSES = ['cargo', 'boat']

    def __init__(self, folder, assets='.', cam_num=4):
        self.classes_involved = None
        self.bod = None
        self.cam_name = None
        self.folder = folder
        self.assets_dir = assets + "/assets/"
        self.s_time = {}
        self.data, self.r_data = None, None
        self.loc, self.time_sequence, self.visible_map = None, None, None
        xls = xlrd.open_workbook_xls(self.assets_dir + 'time/time_seg.xls').sheet_by_index(0)
        i = 1
        while i <= xls.nrows - 1:
            name = str(xls.cell_value(i, 0))
            pre_frame = int(xls.cell_value(i, 1))
            if pre_frame % 5 < 3:
                pre_frame = pre_frame - pre_frame % 5
            else:
                pre_frame = pre_frame - pre_frame % 5 + 5
            t = Time(int(xls.cell_value(i, 2)) * 10000)
            t = Time.time_f_datetime(t.time - timedelta(seconds=pre_frame / 25))
            self.s_time[name] = t

            i += 1

        self.t_vector, self.r_matrix_inv, self.k_matrix_inv = [], [], []
        for i in range(cam_num):
            intrinsic = np.load(self.assets_dir + "intrinsic/cam_{}.npy".format(i + 1))
            r_vector = np.load(self.assets_dir + "rvec/cam_{}.npy".format(i + 1))
            r_matrix = cv2.Rodrigues(r_vector)[0]

            self.t_vector.append(np.load(self.assets_dir + "tvec/cam_{}.npy".format(i + 1)))
            self.r_matrix_inv.append(np.linalg.inv(r_matrix))
            self.k_matrix_inv.append(np.linalg.inv(intrinsic))

    def pixel2world(self, data_path, camera_id, st):

        # ---------------- get parameters ----------------
        t_vector = self.t_vector[camera_id]
        r_matrix_inv = self.r_matrix_inv[camera_id]
        k_matrix_inv = self.k_matrix_inv[camera_id]

        # ---------------- set height equal to 0 ----------------
        data = np.loadtxt(data_path + r'\data.txt', delimiter=',')  # size(nt, 5)
        visible_time = 0
        for i in range(data.shape[0]):
            if data[i, -1] > 0:
                visible_time += 1
        if visible_time > 50:

            with open(data_path + '/cfg.json', 'r', encoding='UTF-8') as f:
                cfg = json.loads(f.read())
            cls = cfg['cls']
            ed = cfg['end_frame']
            obj_id = cfg['id']
            sf = ed - data.shape[0] + 1
            if sf < 0:
                print(data_path)
                raise ValueError('end frame is less than start frame')

            for ed in range(data.shape[0] - 1, -1, -1):

                conf = data[ed][4]
                xy = data[ed][:4]
                if conf > 0:
                    if (xy[2] - 1920) ** 2 > 3 and (xy[3] - 1080) ** 2 > 3 and xy[0] ** 2 > 3 and xy[1] ** 2 > 3:
                        break
                pass

            data = data[:ed]
            world = np.zeros((data.shape[0], 3))
            idx = 0
            res_t = []
            sec = []
            for i in data:
                pixel_x = (i[0] / 3 + i[2] * 2 / 3) * 4 / 3
                pixel_y = (i[1] * 2 / 3 + i[3] / 3) * 4 / 3
                matrix1 = np.dot(np.dot(r_matrix_inv, k_matrix_inv), np.array([pixel_x, pixel_y, 1]).reshape(3, 1))
                matrix2 = np.dot(r_matrix_inv, t_vector.reshape(3, 1))
                temp_height = (0 + matrix2[2]) / matrix1[2]
                world_cal = matrix1 * temp_height - matrix2

                t = st.time + timedelta(seconds=(sf + idx) / 5)
                res_t.append(t)
                t = mktime(
                    time.strptime(t.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')) + t.microsecond / 1000000
                sec.append(t)

                world_cal_xy = [float(world_cal[0]), float(world_cal[1]), i[4]]
                world[idx] = world_cal_xy
                idx += 1

            if data.shape[0] > 0:
                return True, world, (res_t, np.array(sec), cls, obj_id)
            else:
                return False, None, (None, None, None, None)
        else:
            return False, None, (None, None, None, None)

    def scene(self, scene_num):
        targets = []
        data = []
        for file in os.listdir(self.folder):
            if file.startswith('%d' % scene_num):
                targets.append(file)
        for target in targets:
            file = os.path.join(self.folder, target)
            st = self.s_time[target]
            for obj in os.listdir(file):
                if os.path.isdir(os.path.join(file, obj)):
                    flag, world, (res_t, sec, cls, obj_id) = self.pixel2world(os.path.join(file, obj),
                                                                              int(file[-1]) - 1, st)
                    if flag:
                        data.append({'id': obj_id, 'video': target, 'cls': cls, 'time': res_t, 'sec': sec,
                                     'data': world, 'conf': world[:, 2]})

        classes_involved = []
        for obj in data:
            if obj['cls'] not in classes_involved:
                classes_involved.append(obj['cls'])
        self.classes_involved = classes_involved
        self.data = data
        del data
        gc.collect()

        for cls in self.classes_involved:
            self.data_reg(cls)
            self.world_track()

    def data_reg(self, cls):

        st = 1e11
        ed = 0
        x_min = 1e9
        x_max = -1e9
        y_min = 1e9
        y_max = -1e9
        data = []
        for obj in self.data:
            if obj['cls'] == cls:
                data.append(obj)
                if st > obj['sec'].min():
                    st = obj['sec'].min()
                if ed < obj['sec'].max():
                    ed = obj['sec'].max()
                if x_min > obj['data'][:, 0].min():
                    x_min = obj['data'][:, 0].min()
                if x_max < obj['data'][:, 0].max():
                    x_max = obj['data'][:, 0].max()
                if y_min > obj['data'][:, 1].min():
                    y_min = obj['data'][:, 1].min()
                if y_max < obj['data'][:, 1].max():
                    y_max = obj['data'][:, 1].max()

        for obj in data:
            obj['sec'] = np.around((obj['sec'] - st) / 2, decimals=1)
            obj['data'] = np.hstack([obj['data'][:, :2], obj['sec'].reshape(-1, 1)])

        cam_name = []
        for obj in data:
            if obj['video'] not in cam_name:
                cam_name.append(obj['video'])

        self.cam_name = []
        for i in [0, 1, 2, 3]:
            for cam in cam_name:
                if cam.endswith(str(i + 1)):
                    self.cam_name.append(cam_name[i])
        self.cam_name.reverse()

        self.bod = [[x_min, x_max], [y_min, y_max]]

        self.r_data = {i: [] for i in cam_name}
        for obj in data:
            self.r_data[obj['video']].append(obj)

        self.time_sequence = np.arange(0, np.around((ed - st) * 5, decimals=0) + 0.2, step=0.2)

    def plot_all(self, i=-1):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        plt.xlim(self.bod[0])
        plt.ylim(self.bod[1])
        if i == -1:
            for i in range(len(self.cam_name)):
                cam = self.cam_name[i]
                for obj in self.r_data[cam]:
                    ax.scatter(obj['data'][:, 0], obj['data'][:, 1], obj['data'][:, 2], c=COLOR_MAP[i], s=2,
                               label=obj['video'])
        else:
            cam = self.cam_name[i]
            for i in range(len(self.r_data[cam])):
                obj = self.r_data[cam][i]
                ax.scatter(obj['data'][:, 0], obj['data'][:, 1], obj['data'][:, 2], c=COLOR_MAP[i], s=2,
                           label=obj['video'])
        ax.legend()
        plt.show()

    def world_track(self):
        # self.plot_all(-1)
        for i in range(len(self.cam_name)):
            while True:
                if self.connect(self.cam_name[i]):
                    break
        # self.plot_all(-1)
        for i in range(2):
            for j in range(1, 3):
                self.try_bind(self.cam_name[i], self.cam_name[j])

    def connect(self, cam1, cam2=None):
        cam_objs1 = self.r_data[cam1]
        if cam2 is None:
            cam_objs2 = cam_objs1
        else:
            cam_objs2 = self.r_data[cam2]
        st_point, ed_point = [], []
        for obj in cam_objs2:
            st_point.append(obj['data'][0])
        for obj in cam_objs1:
            ed_point.append(obj['data'][-1])
        for i in range(len(ed_point)):
            for j in range(len(st_point)):
                if i != j and st_point[j][2] + 2 > ed_point[i][2]:
                    f1 = self.k_factor(cam_objs1[i])
                    p_xy = [st_point[j][2] * f1[0][0] + f1[0][1], st_point[j][2] * f1[1][0] + f1[1][1]]
                    a_xy = st_point[j][:2]
                    n_f1 = np.array(f1)[:, 0]
                    n_f2 = np.array(self.k_factor(cam_objs2[j]))[:, 0]
                    angle = np.arccos(np.dot(n_f1, n_f2) / np.sqrt(np.dot(n_f1, n_f1)) / np.sqrt(np.dot(n_f2, n_f2)))
                    accuracy = np.linalg.norm(np.array(p_xy) - np.array(a_xy), ord=2) * (angle ** 0.5)
                    dt = max(np.abs(st_point[j][2] + 2 - ed_point[i][2]), 2)
                    accuracy = accuracy / (dt ** 0.5)
                    print(cam1, accuracy)
                    if accuracy < 15:
                        data = cam_objs1[i]['data']
                        conf = cam_objs1[i]['conf']
                        new_st = st_point[i][2]
                        new_ed = ed_point[j][2]
                        cam_objs1[i]['sec'] = np.around(np.arange(new_st, new_ed + 0.1, step=0.1), decimals=1)
                        cam_objs1[i]['data'] = np.zeros((cam_objs1[i]['sec'].shape[0], 3))
                        d_st = data.shape[0]
                        cam_objs1[i]['data'][:d_st, :] = data
                        cam_objs1[i]['conf'] = - np.ones((cam_objs1[i]['sec'].shape[0],))
                        cam_objs1[i]['conf'][:conf.shape[0]] = conf
                        data = cam_objs2[j]['data']
                        conf = cam_objs2[j]['conf']
                        d_fn = cam_objs1[i]['sec'].shape[0] - data.shape[0]
                        cam_objs1[i]['data'][d_fn:, :] = data
                        cam_objs1[i]['conf'][-conf.shape[0]:] = conf
                        cam_objs1[i]['data'][:, 2] = cam_objs1[i]['sec']
                        cam_objs1[i]['data'][d_st: d_fn, 0] = np.interp(cam_objs1[i]['data'][d_st: d_fn, 2],
                                                                        [cam_objs1[i]['data'][d_st - 1][2],
                                                                         cam_objs1[i]['data'][d_fn][2]],
                                                                        [cam_objs1[i]['data'][d_st - 1][0],
                                                                         cam_objs1[i]['data'][d_fn][0]])
                        cam_objs1[i]['data'][d_st: d_fn, 1] = np.interp(cam_objs1[i]['data'][d_st: d_fn, 2],
                                                                        [cam_objs1[i]['data'][d_st - 1][2],
                                                                         cam_objs1[i]['data'][d_fn][2]],
                                                                        [cam_objs1[i]['data'][d_st - 1][1],
                                                                         cam_objs1[i]['data'][d_fn][1]])
                        if cam2 is None:
                            del cam_objs2[j]
                        return False
        return True

    def try_bind(self, cam1, cam2):
        cam_objs1 = self.r_data[cam1]
        cam_objs2 = self.r_data[cam2]
        bind_map = []
        for i in range(len(cam_objs1)):
            bind_map.append([])
            for j in range(len(cam_objs2)):
                # 判断是否时间上重叠
                t1 = (cam_objs1[i]['sec'][0], cam_objs1[i]['sec'][-1])
                t2 = (cam_objs2[j]['sec'][0], cam_objs2[j]['sec'][-1])
                sp1 = self.speed(cam_objs1[i])
                sp2 = self.speed(cam_objs2[j])
                f1 = np.array(self.k_factor(cam_objs1[i]))
                n_f1 = f1[:, 0]
                f2 = np.array(self.k_factor(cam_objs2[j]))
                n_f2 = f2[:, 0]
                angle = np.arccos(np.dot(n_f1, n_f2) / np.sqrt(np.dot(n_f1, n_f1)) / np.sqrt(np.dot(n_f2, n_f2)))
                bind_map[i].append({'state': 's', 'vector': (angle, sp1, sp2), 'dis': None})
                if t1[0] > t2[1] or t1[1] < t2[0]:
                    if t1[0] > t2[1]:
                        p_t1 = t1[0]
                        p_t2 = t2[1]
                        p_i1 = 0
                        p_i2 = -1
                    else:
                        p_t1 = t1[1]
                        p_t2 = t2[0]
                        p_i1 = -1
                        p_i2 = 0
                    p_xy1 = [p_t2 * f1[0][0] + f1[0][1], p_t2 * f1[1][0] + f1[1][1]]
                    a_xy1 = cam_objs2[j]['data'][p_i2][:2]
                    p_xy2 = [p_t1 * f2[0][0] + f2[0][1], p_t1 * f2[1][0] + f2[1][1]]
                    a_xy2 = cam_objs1[i]['data'][p_i1][:2]
                    dis1 = np.linalg.norm(np.array(p_xy1) - np.array(a_xy1))
                    dis2 = np.linalg.norm(np.array(p_xy2) - np.array(a_xy2))
                    bind_map[i][j]['dis'] = [dis1, dis2]
                    pass
                else:
                    st = max(t1[0], t2[0])
                    ed = min(t1[1], t2[1])
                    si = int(np.around((st-t1[0])*10, decimals=0))
                    ei = int(np.around((ed-st)*10, decimals=0)) + si
                    data1 = cam_objs1[i]['data'][si:ei]
                    si = int(np.around((st-t2[0])*10, decimals=0))
                    ei = int(np.around((ed-st)*10, decimals=0)) + si
                    data2 = cam_objs2[j]['data'][si:ei]
                    dis, std = self.distance(data1, data2)
                    bind_map[i][j]['dis'] = [dis, std]
                    bind_map[i][j]['state'] = 'o'
        return bind_map

    @staticmethod
    def k_factor(obj):
        reg = LinearRegression()
        x = obj['data'][:, 2].reshape(-1, 1)
        y = obj['data'][:, 0].reshape(-1, 1)
        y, _ = Boat.kalman(y[::-1], r=0.01)
        y = y[::-1]
        reg.fit(x, y)
        res = [[reg.coef_[0][0], reg.intercept_[0]]]
        y = obj['data'][:, 1].reshape(-1, 1)
        y, _ = Boat.kalman(y[::-1], r=0.01)
        y = y[::-1]
        reg.fit(x, y)
        res.append([reg.coef_[0][0], reg.intercept_[0]])
        return res

    @staticmethod
    def speed(obj):
        x = obj['data'][:, 0].reshape(-1, 1)
        x, _ = Boat.kalman(x[::-1], r=0.01)
        y = obj['data'][:, 1].reshape(-1, 1)
        y, _ = Boat.kalman(y[::-1], r=0.01)
        xys = np.vstack((x, y))
        delta = 0
        for xy1, xy2 in zip(xys[:-1], xys[1:]):
            delta += np.linalg.norm(xy1 - xy2, ord=2)
        delta = delta / xys.shape[0]
        return delta

    @staticmethod
    def distance(data1, data2):
        data1, _ = Boat.kalman(data1[:, :2], r=0.01)
        data2, _ = Boat.kalman(data2[:, :2], r=0.01)
        dis = 0
        diss = []
        for xy1, xy2 in zip(data1, data2):
            dis += np.linalg.norm(xy1 - xy2, ord=2)
            diss.append(dis)
        dis = dis / data1.shape[0]
        std = np.std(np.array(diss)) / data1.shape[0]
        return dis, std


def main():
    folder = r'E:\AICE\ICSHM2022\Pj_1\trace_result'
    mp = MAP(folder)
    scene_num = [7]
    # scene_num = [1, 2, 3, 4, 5, 6, 7]
    for i in scene_num:
        mp.scene(i)


if __name__ == '__main__':
    main()
