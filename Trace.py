"""
结果保存在trace_result下
目录
    cfg.json  {"id": 船只id(int), "cls": 船只类型(str), "end_frame": 船只最后一次出现时的视频帧数(int)}
    data.npy  np.array([[x1, y1, x2, y2, confidence]...])  按时间顺序正序排列，船只消失时confidence=-1
"""

import os
from collections import deque
import cv2
import matplotlib
import numpy as np
import torch
import torchvision
from sklearn.linear_model import LinearRegression
import json
from concatenate import cct

matplotlib.use('TkAgg')
HISTORY_LEN = 200
COLOR_MAP = ['red', 'orange', 'yellow', 'lime', 'aqua', 'blue', 'fuchsia',
             'darkred', 'peru', 'darkgoldenrod', 'darkgreen', 'teal', 'darkblue', 'purple',
             'coral', 'khaki', 'mediumaquamarine', 'dodgerblue', 'mediumslateblue', 'deeppink']
LABEL_MAP = ['cargo', 'boat', 'buoy', 'bridge']


class Trace:
    """
    Trace 视频追踪结果读取，存储，处理

    Args:
        fps (int): 视频的实际帧率

    Attributes:
        fps (int): 视频的实际帧率
        id_max (int): 最大id
        len (int): 总帧数
        model (YOLO.model)
        nms_lock (list[int]): 无论是否开启nms，里面的id都会执行nms，用于处理不追踪物体
        height (int): 图片高度
        width (int): 图片宽度
        id (list(int)): 每帧所有出现目标的id, 不追踪目标为-1
        xy (list): 每帧所有出现目标的检测框xy size: (frames, objects, 4)
        conf (list): 每帧所有出现目标的类和置信度 size: (frames, objects, 2)

    Methods:
        save: 存储到文件夹
        load: 加载文件夹内信息
        nxy: 输出box的中心坐标(nx, ny)
        __plot: 调用yolo result的plot
        animate:  输出动画
        predict: 预测单一图片
        track: 追踪视频

    Returns:
    """

    # noinspection PyTypeChecker
    def __init__(self, fps: int = 1, model=None) -> None:
        self.fps = fps
        self.id_max = 0
        self.model = model
        self.nms_lock = [2, 3]  # default
        self.len = 0
        # 读取结果
        self.height = 0
        self.width = 0
        self.id, self.xy, self.conf, self.cls = [], [], [], []

        self.track_his = None

    def r_plot(self, pic, wait_key=20, thickness=2, output=None, plot=True):  # text: (内容， 坐标[x, y])
        # cv2.rectangle(img, pt1, pt2, color, thickness, lineType, shift)
        # cv2.putText(image, text, org, font, fontScale, color, thickness, lineType, bottomLeftOrigin)
        res_plotted = cv2.imread(pic)
        # 将各个目标plot到图片上
        for idx, cls, xy, conf, vector in \
                (zip(self.id, self.cls, self.xy, self.conf, self.xy) if output is None else output):
            pt1, pt2 = [int(xy[0]), int(xy[1])], [int(xy[2]), int(xy[3])]
            cls = int(cls)
            color = matplotlib.colors.to_rgb(COLOR_MAP[(3*idx + 7) % len(COLOR_MAP)])
            color = (color[2] * 255, color[1] * 255, color[0] * 255)
            cv2.rectangle(res_plotted, pt1, pt2, color, thickness)  # 目标框
            cv2.rectangle(res_plotted, pt1, [pt1[0] + 85, pt1[1] - 14], color, thickness=-1)  # 左下角小矩形显示信息
            if idx < 0:
                text = LABEL_MAP[cls] + '  %0.2f' % conf
            else:
                text = 'id%d %s %0.2f' % (idx, LABEL_MAP[cls], conf)
                cp = [int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2)]
                vector = np.array(vector)

                if np.isnan(vector[0]) or np.isnan(vector[1]):
                    vector = np.array([0, 0])
                elif 0 < np.linalg.norm(vector) < 3:
                    vector = vector / np.linalg.norm(vector) * 3
                vp = [int(cp[0] + 4 * vector[0]), int(cp[1] + 4 * vector[1])]
                cv2.arrowedLine(res_plotted, cp, vp, color, thickness=2, tipLength=0.15, line_type=cv2.LINE_AA)
            cv2.putText(res_plotted, text, [pt1[0] + 1, pt1[1] - 3], cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        if plot:
            cv2.imshow("result", res_plotted)
            cv2.waitKey(wait_key)
        return res_plotted

    def predict(self, source=None, nms_disable=False, confidence=0.3, plot=False, wait_key=0, save=False, **kwargs):
        # source: 图片
        # nms_disable: 是否启用nms
        # plot: 图片
        if self.model is None:
            name = self.__class__.__name__
            raise AttributeError(f''''{name}' object has no attribute 'model', model must be resign first.''')
        result = self.model.predict(source=source, nms_disable=nms_disable, **kwargs)[0].cpu()

        # 对结果中被nms_lock指定的id进行nms，替换原结果
        xy, conf, cls = result.boxes.xyxy, result.boxes.conf, result.boxes.cls
        if cls.shape[0] != 0:
            for i in range(cls.amax().int() + 1):
                if (i in self.nms_lock) and i in cls:  # 判断需要nms
                    chose = cls == i
                    idx_list = torch.arange(conf.shape[0])[chose]  # 被选中目标的id
                    conf_s, xy_s = conf[chose], xy[chose]  # 筛选目标
                    nms_id = torchvision.ops.nms(xy_s, conf_s, confidence)  # NMS
                    chose = cls != i
                    chose[idx_list[nms_id]] = True  # 去除大部分该id的box，保留了一部分
                    xy, conf, cls = xy[chose], conf[chose], cls[chose]

        if plot:
            self.xy, self.conf, self.cls = xy.tolist(), conf.tolist(), cls.tolist()
            self.id = [-1 for _ in range(len(self.cls))]
            plot = self.r_plot(source, wait_key=wait_key)
            if save:
                cv2.imwrite('plot.jpf', plot)

        return xy, conf, cls, result.orig_shape

    def predict_video(self, folder, nms_disable=True, confidence=0.3, stream=False, wait_key=20, reverse=False,
                      save_video=False, st=0, **kwargs):
        list_pic = listdir_abs(folder)[st:]
        four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 设置输出视频为mp4格式
        if reverse:
            list_pic.reverse()
        video, save_dir, num = None, None, 0
        pj_name = folder[:folder.rfind('.mp4')]
        pj_name = pj_name[pj_name.rfind('\\')+1:]
        print('\n---Predicting video %s' % pj_name)

        for i in range(len(list_pic)):
            pic = list_pic[i]
            print('\rPredicting frame %d / %d' % (i+1, len(list_pic)), end='', flush=True)
            xy, conf, cls, shape = self.predict(pic, nms_disable=nms_disable, confidence=confidence, plot=False,
                                                wait_key=20, **kwargs)
            self.width, self.height = shape[1], shape[0]

            if i == 0 or self.track_his is None:
                self.track_his = TrackHistory(self.width, self.height, project_name=pj_name)
                save_dir = self.track_his.save_dir

            output = self.track_his.next_frame((xy, conf, cls))
            if save_video:
                img = self.r_plot(pic, wait_key=wait_key, output=output, plot=stream)
                if save_video:
                    if i % 1000 == 0 and video is not None:
                        video.release()
                        video = None
                        num += 1
                    if video is None:  # 新建视频
                        v_name = save_dir + '\\predict_%d.mp4' % num
                        if os.path.isfile(v_name):
                            os.remove(v_name)
                        video = cv2.VideoWriter(v_name, four_cc, fps=25, frameSize=[self.width, self.height])
                    video.write(img)
            elif stream:
                self.r_plot(pic, wait_key=wait_key, output=output)

        print('')
        if video is not None:
            video.release()
            cct(save_dir)
        print('predict done')


class TrackHistory:
    """
    Track_history 追踪历史

    Args:
        length(int): 历史保留长度

    Attributes:
        len(int): 历史保留长度
        width(int): 图片尺寸
        height(int): 图片尺寸
        world_state(str): 'No ship', 'Some ship', 'Hiding ship'
        world_state_next(int): 一帧开始时的world_state index
        target_label : 需要追踪的label id
        boxes_in: 输入的boxes信息
        conf_default: 对船类目标进行nms时的conf
        frames: 当前帧数

    Methods:
        update(obj: list, xy: list): 在追踪完成后刷新已有记录
        next_frame(result: yolo result): 输入下一帧的预测结果，开始新一轮追踪
        wait_ship(): world_state为 no ship时，等待新的船只进入
        _visible_obj(): 返回所有可见obj的id及该目标物可见的时间长度

    Returns:
    """

    STATE_MAP = ['No ship', 'Some ship', 'Hiding ship', 'clear']

    def __init__(self, width, height, length=HISTORY_LEN, project_name=None):
        self.len = length
        self.width = width
        self.height = height
        self.obj_next = 0  # 下一个目标的编号
        self.world_state = self.STATE_MAP[0]
        self.world_state_next = 1
        self.target_label = [0, 1]  # default
        self.boxes_in = None
        self.conf_default = 0.2
        self.boat = []
        self.cross_obj = deque([])
        self.frames = -1

        if not os.path.exists(r'.\trace_result'):  # 主文件夹
            os.makedirs(r'.\trace_result')

        # 本次结果文件夹
        if project_name is not None:
            self.save_dir = r'.\trace_result\%s' % project_name
        else:
            save_id = 0
            save_path = '.\\trace_result\\trace_result'
            while True:
                if not os.path.exists(save_path + '%02d' % save_id) or save_id > 99:
                    break
                save_id += 1
            self.save_dir = save_path + '%02d' % save_id
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        for file in listdir_abs(self.save_dir):
            if os.path.isdir(file):
                if os.path.isfile(file + r'\data.txt'):
                    os.remove(file+r'\data.txt')

    def next_frame(self, result):

        # 读取参数到boxes，去掉桥和浮标
        f = False
        _, _, cls = result
        chose = cls == self.target_label[0]
        chose_t = cls == self.target_label[1]
        chose = chose | chose_t
        self.boxes_in = Boxes(result, chose)  # 待处理的boxes
        chose = chose == f
        t_boxes = Boxes(result, chose)
        res = []
        for i in range(t_boxes.xy.shape[0]):
            res.append([-1, int(t_boxes.cls[i]), t_boxes.xy[i].tolist(), float(t_boxes.conf[i]), [0, 0]])
        self.frames += 1

        # 追踪顺序：先处理碰撞遮挡区域，再处理原船只运行区域，最后还有剩余的box生成新的追踪目标
        while self.world_state != 'clear':
            if self.world_state == 'Hiding ship':
                self.predict_ship()
            elif self.world_state == 'Some ship':
                self.track_ship()
            else:
                self.wait_ship()

        self.world_state = self.STATE_MAP[self.world_state_next]  # 更新世界状态
        self.world_state_next = 0  # 重置下一帧世界状态
        self.boat_refresh()

        self.save_trace()
        for boat in self.boat:
            if boat.plot:
                res.append((boat.info()))
        return res

    def wait_ship(self):
        # 无船状态下，出现可能的新船都被认为是一个新目标
        if self.boxes_in.xy.shape[0] != 0:
            nms_id = torchvision.ops.nms(self.boxes_in.xy, self.boxes_in.conf, self.conf_default)  # NMS
            for i in range(nms_id.shape[0]):
                #  添加新的目标
                if float(self.boxes_in.conf[nms_id[i]]) < 0.6  :  # 置信度大于某个值才能被认为是新的船，之后追踪不受影响
                    continue
                new_obj = Boat(idx=self.obj_next, cls=self.boxes_in.cls[nms_id[i]], xy=self.boxes_in.xy[nms_id[i]],
                               shape=[self.width, self.height], conf=self.boxes_in.conf[nms_id[i]])
                self.boat.append(new_obj)
                self.obj_next += 1

        self.world_state = self.STATE_MAP[3]  # 下一帧
        self.world_state_next = max(1, self.world_state_next)  # 若world_state_next已被更改为2, 则无法通过此方法变为1

    def track_ship(self):
        box_pre, utc, obj = [], [], []
        for boat in self.boat:  # 对所有的船
            if boat.track():  # 可以追踪
                b_pre, u_pre = boat.b_predict()
                box_pre.append(b_pre), utc.append(u_pre), obj.append(boat.idx)

        dis_idx = self.boxes_in.distribute(box_pre, utc)
        for i in range(len(obj)):  # 对所有正在追踪的船更新nms结果
            if len(dis_idx[i]) == 0:  # 没有相似的目标
                self.boat[obj[i]].update((box_pre[i], -1, -1), state=1)  # 当作暂时消失
            else:
                self.boat[obj[i]].update(self.boxes_in.b_nms(dis_idx[i], self.conf_default), box_pre[i], state=0)

        #  删掉已被检测的船
        idx_all = []
        for idx in dis_idx:
            idx_all.extend(idx)
        self.boxes_in.drop(idx_all)
        for i in range(len(obj)):
            idx, _, _ = self.boxes_in.is_in(self.boat[obj[i]].xy[0], expand=30)
            self.boxes_in.drop(idx)

        if self.boxes_in.xy.shape[0] != 0:  # 做完追踪后还有剩余boxes
            self.world_state = self.STATE_MAP[0]  # 按新出现船处理
        else:
            self.world_state = self.STATE_MAP[3]  # 下一帧
        self.world_state_next = max(1, self.world_state_next)  # 若world_state_next已被更改为2, 则无法通过此方法变为1

    def predict_ship(self):
        def p_range(co_in1, co_in2, expand=0):  # 返回两个矩形的最小包络矩形
            xs = [co_in1[0], co_in1[2], co_in2[0], co_in2[2]]
            ys = [co_in1[1], co_in1[3], co_in2[1], co_in2[3]]
            return [min(xs) - expand, min(ys) - expand, max(xs) + expand, max(ys) + expand]

        def offset(co_xy, t_range, x_drc, y_drc):
            def __move(move_co_xy, move_d, move_drc):
                if move_drc == 0:
                    return [move_co_xy[0] + move_d, move_co_xy[1], move_co_xy[2] + move_d, move_co_xy[3]]
                else:
                    return [move_co_xy[0], move_co_xy[1] + move_d, move_co_xy[2], move_co_xy[3] + move_d]

            bs = t_range[x_drc] - co_xy[x_drc]
            co_xy = __move(co_xy, bs, 0)
            bs = t_range[y_drc] - co_xy[y_drc]
            co_xy = __move(co_xy, bs, 1)
            return co_xy

        pops = []  # 要取消的追踪
        for i in range(len(self.cross_obj)):
            obj1, obj2 = self.cross_obj[i]
            boat1, boat2 = self.boat[obj1], self.boat[obj2]
            if boat1.gone or boat2.gone:
                boat1.cross = False
                boat2.cross = False
                pops.append(i)
                self.world_state = self.STATE_MAP[1]
                continue
            pre1, utc1 = boat1.b_predict()
            pre2, utc2 = boat2.b_predict()
            ct1 = [(pre1[0] + pre1[2]) / 2, (pre1[1] + pre1[3]) / 2]
            ct2 = [(pre2[0] + pre2[2]) / 2, (pre2[1] + pre2[3]) / 2]

            # 预测框偏移的方向
            drc1, drc2 = [], []
            if ct1[0] > ct2[0]:  # 1在2右边
                drc1.append(2), drc2.append(0)  # 1右2左
                boat1.side, boat2.side = 0, 2
            else:
                drc1.append(0), drc2.append(2)  # 1左2右
                boat1.side, boat2.side = 2, 0
            if ct1[1] > ct2[1]:  # 1在2上边
                drc1.append(3), drc2.append(1)
            else:
                drc1.append(1), drc2.append(3)
            act_range = p_range(boat1.xy[0], boat2.xy[0])  # 实际范围
            pre_range = p_range(pre1, pre2)  # 预测范围
            abv_range = p_range(act_range, pre_range, 100)  # 范围的范围

            dis_idx1, dis_idx2 = self.boxes_in.distribute([pre1, pre2], [utc1 * 2, utc2 * 2])  # 不确定性增加
            if len(dis_idx1) != 0:
                boat1.update(self.boxes_in.b_nms(dis_idx1, self.conf_default), pre1, state=0)
            else:
                boat1.update([offset(pre1, act_range, drc1[0], drc1[1]), -1, -1], pre1, state=1)
            if len(dis_idx2) != 0:
                boat2.update(self.boxes_in.b_nms(dis_idx2, self.conf_default), pre2, state=0)
            else:
                boat2.update([offset(pre2, act_range, drc2[0], drc2[1]), -1, -1], pre2, state=1)
            dis_idx1.extend(dis_idx2)
            self.boxes_in.drop(dis_idx1)
            contain_idx, _, _ = self.boxes_in.is_in(abv_range)
            self.boxes_in.drop(contain_idx)

            self.world_state_next = 2  # 接下来仍在相遇过程中
            if not self.intercept(pre1, pre2):  # 两船下一时刻不相交
                boat1.cross, boat2.cross = False, False
                boat1.side, boat2.side = -1, -1
                pops.append(i)

            self.world_state = self.STATE_MAP[1]  # 进行追踪

        pops.reverse()  # 从大到小
        for i in pops:  # 删掉已经解除的追踪
            del self.cross_obj[i]
        if len(self.cross_obj) == 0:  # 没有相遇事件
            self.world_state_next = 1  # 回到追踪状态
        pass

    def boat_refresh(self):  # 检查船只状态并刷新

        #  刷新船的状态
        del_idx = []
        for i in range(len(self.boat)):
            if self.boat[i].gone or (i in del_idx):  # 跳过
                continue
            boat = self.boat[i]
            if not boat.plot:
                if boat.state[0] == 0:  # 重新出现
                    boat.plot = True
            if boat.state[0] != 0:
                if boat.state.index(0) >= 100:  # 消失超过一百帧
                    boat.gone = True
                    boat.plot = False
                    continue
                if (boat.xy[0][0] > self.width - 10) or (boat.xy[0][2] < 10):  # 在船边缘距离左右边缘30像素以内时消失
                    boat.gone = True
                    boat.plot = False
                    continue
                if boat.state.index(0) >= 10 and boat.state[0] == 1:  # 暂时消失
                    boat.plot = False
            if boat.track():
                for j in range(i + 1, len(self.boat)):
                    if self.boat[j].gone:
                        continue
                    if self.intercept(boat.future(), self.boat[j].future()):
                        if len(self.boat[j]) < 5:  # 刚出现就有碰撞
                            del_idx.append(j)
                            continue
                        self.boat[i].cross, self.boat[j].cross = True, True
                        if boat.xy[0][0] < self.boat[j].xy[0][0]:  # 在左边
                            self.boat[i].side, self.boat[j].side = 2, 0
                        else:
                            self.boat[i].side, self.boat[j].side = 0, 2
                        self.cross_obj.append([self.boat[j].idx, self.boat[i].idx])
                        self.world_state = self.STATE_MAP[2]
            if len(boat.state) > 1:
                if boat.state[0] == 0 and boat.state[1] != 0:
                    boat.inter_p()
        del_idx = list(set(del_idx))
        while len(del_idx) > 0:
            idx = max(del_idx)
            del self.boat[idx]
            del_idx.remove(idx)

        id_next = 0
        for boat in self.boat:
            id_next = max(id_next, boat.idx + 1)
        self.obj_next = id_next

    @staticmethod
    def intercept(xy1, xy2):
        def contain(a1, a2, b1, b2):
            if b1 > a2 or a1 > b2:
                return False
            else:
                return True

        if contain(xy1[0], xy1[2], xy2[0], xy2[2]) and contain(xy1[1], xy1[3], xy2[1], xy2[3]):
            return True
        else:
            return False

    def save_trace(self):  # 保存船只追踪数据

        for boat in self.boat:
            if boat.gone:
                continue
            save_dir = self.save_dir + '\\object%02d' % boat.idx  # 文件夹
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            cfg = {'id': boat.idx, 'cls': LABEL_MAP[boat.cls], 'end_frame': self.frames}
            with open(save_dir + '\\cfg.json', 'w') as fp:
                json.dump(cfg, fp)

            file_name = save_dir + '\\data.txt'
            if not os.path.isfile(file_name):
                with open(file_name, 'w') as fp:
                    fp.write('')

            with open(file_name, 'a') as fp:
                for i in range(4):
                    fp.write('%.4f, ' % boat.xy[0][i])
                fp.write('%.4f\n' % boat.conf[0])


class Boat:
    PREDICT_LEN = 20  # 用于预测下一帧位置的帧数
    UCT_BIAS = 25  # 基础的不确定度
    UCT_DELTA = 2  # 不确定度递增系数
    WARM = 50

    def __init__(self, idx: int, cls, xy, conf, shape, length=HISTORY_LEN):
        self.boundary = shape
        self.idx = idx
        self.cls = int(cls)
        self.vector = np.array([0, 0])
        self.xy, self.conf, self.state = deque(maxlen=length), deque(maxlen=length), deque(maxlen=length)
        self.cls_certainty = 0
        self.coe = deque(maxlen=length)
        self.coe.appendleft(np.array([0, 0, 0, 0], dtype=float))
        self.xy.appendleft(np.array(xy, dtype=float))
        self.conf.appendleft(float(conf))
        self.state.appendleft(0)  # 0为可见，1为暂时消失
        self.gone = False  # 判定为离开，不再追踪
        self.side = -1  # 0在碰左边，2在碰右边
        self.plot = True  # 是否显示
        self.cross = False  # 是否相遇

    def b_predict(self):  # 返回预测位置
        #  计算预测帧数
        xy, x = [], []
        m_len = min(self.PREDICT_LEN, len(self.state))
        i = 0

        s_xy, _ = self.kalman(np.array(self.xy))
        while True:
            if (i > m_len and len(x) > 5) or i >= len(self):
                break
            if self.state[i] == 0:
                xy.append(s_xy[i])
                x.append([i])
            i += 1
        xy = np.array(xy)[::-1]
        x = np.array(x)[::-1]

        xy_pre = []
        coe = []
        reg = LinearRegression()
        for i in range(4):
            if x.shape[0] != 0:
                reg.fit(x, xy[:, i])
                coe.append(-reg.coef_[0])
            else:
                coe.append(0)
            xy_pre.append(float(coe[i] + self.xy[0][i]))  # 预测值

        self.coe.appendleft(np.array(coe))

        vx = (coe[0] + coe[2])
        vy = (coe[1] + coe[3])
        self.vector = self.vector*0.7 + np.array([vx, vy])*0.3

        utc = self.UCT_BIAS + self.UCT_DELTA * (self.PREDICT_LEN - self.visible_time())
        if LABEL_MAP[self.cls] == 'cargo':  # 属于货船
            utc = utc * max(self.size() / 10000, 1)  # 根据船只面积改变不确定度
        else:
            utc = utc * max(self.size() / 4000, 1)   # 根据船只面积改变不确定度
        return xy_pre, utc

    @ staticmethod
    def kalman(in_z: np.ndarray, q=1e-5, r=0.01 ** 2):

        res_z = np.zeros_like(in_z)
        res_cov = np.zeros_like(in_z)
        for i in range(in_z.shape[1]):
            z = in_z[:, i]
            n_iter = z.shape[0]
            sz = (n_iter,)
            x_pre = np.zeros(sz)  # x 滤波估计值
            cov = np.zeros(sz)  # 滤波估计协方差矩阵
            x_pre_m = np.zeros(sz)  # x 估计值
            cov_m = np.zeros(sz)  # 估计协方差矩阵
            mat_k = np.zeros(sz)  # 卡尔曼增益

            # initial guesses
            x_pre[0] = z[0]
            cov[0] = 1.0

            for k in range(1, n_iter):
                # 预测
                x_pre_m[k] = x_pre[k - 1]  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
                cov_m[k] = cov[k - 1] + q  # cov(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1

                # 更新
                mat_k[k] = cov_m[k] / (cov_m[k] + r)  # Kg(k)=cov(k|k-1)H'/[HP(k|k-1)H' + R],H=1
                x_pre[k] = x_pre_m[k] + mat_k[k] * (z[k] - x_pre_m[k])
                cov[k] = (1 - mat_k[k]) * cov_m[k]  # cov(k|k) = (1 - Kg(k)H)cov(k|k-1), H=1
            res_z[:, i] = x_pre
            res_cov[:, i] = cov

        return res_z, res_cov

    def visible_time(self,):
        ml = 0
        m_len = min(self.PREDICT_LEN, len(self.state))
        for i in range(m_len):
            if self.state[i] == 0:
                ml += 1
        if ml == 0:
            ml = 1
        return float(ml)

    def future(self):
        xy, v = self.xy[0], self.vector / 2
        xy = [xy[0] + v[0], xy[1] + v[1], xy[2] + v[0], xy[3] + v[1]]
        return xy

    def track(self):
        if self.gone:
            return False
        if (self.state[0] in [0, 1]) and (not self.cross):
            return True
        else:
            return False

    def __len__(self):  # 长度定义为self.state长度
        return len(self.state)

    def update(self, res, pre=None, state=0):

        if self.gone:
            raise IndexError('Boat is gone')

        xy, conf, cls = res
        if int(cls) >= 0:  # 识别到了类别
            if self.cls != int(cls):  # 类别不对
                self.cls_certainty -= 3
                if self.cls_certainty <= 0:  # 更改类别
                    self.cls_certainty = 0
                    self.cls = int(cls)
            else:
                self.cls_certainty += 1  # 类别一致

        if pre is not None:
            # 限制二阶导
            dx1 = xy[0] - pre[0]
            dy1 = xy[1] - pre[1]
            dx2 = xy[2] - pre[2]
            dy2 = xy[3] - pre[3]
            coe = np.array([dx1, dy1, dx2, dy2])
            limit = [4, 1]
            if len(self.coe) > 2:
                for i in range(4):
                    if len(self) < self.WARM:  # 刚出现
                        lim = limit[i % 2] * 10 * self.WARM / len(self)
                    else:
                        lim = limit[i % 2] / np.sqrt(self.visible_time())

                    if self.state[0] == 0:   # 可见
                        lim = lim * 5
                        if len(self) > 1 and self.state[1] == 1:  # 之前消失
                            lim = lim * 25

                    if np.abs(coe[i]) > lim:
                        if coe[i] > 0:
                            coe[i] = lim
                        else:
                            coe[i] = - lim
                if self.side != -1:  # 0在碰左边，2在碰右边
                    coe[self.side] = coe[self.side] / 25
                for i in range(4):
                    xy[i] = float(coe[i] + pre[i])  # 调整值
        xy = np.array([min(xy[0], xy[2]), min(xy[1], xy[3]), max(xy[0], xy[2]), max(xy[1], xy[3])], dtype=float)
        self.xy.appendleft(xy)
        self.conf.appendleft(float(conf))
        self.state.appendleft(state)  # 0为可见，1为遮挡

    def inter_p(self):  # 对两个可见距离内做插值
        x, y = [[0]], [self.xy[0]]
        i = 1
        while i < len(self.state):
            if self.state[i] == 0:
                x.append([i])
                y.append(self.xy[i])
                if len(x) == self.PREDICT_LEN:
                    break
            i += 1

        y = np.array(y)
        reg = LinearRegression()
        coe, itc = [], []
        for i in range(4):
            reg.fit(x, y[:, i])
            coe.append(reg.coef_), itc.append(reg.intercept_)

        cal_x = np.array([i for i in range(x[1][0])])
        cal_xy = np.zeros([4, x[1][0]])
        for i in range(4):
            cal_xy[i] = cal_x * coe[i] + itc[i]
        for i in range(cal_x.shape[0]):
            self.xy[i] = cal_xy[:, i]

    def size(self):
        xy = self.xy[0]
        return (xy[2] - xy[0]) * (xy[3] - xy[1])

    def info(self):
        xy = self.xy[0]
        for i in [0, 1]:
            if xy[i] < 0:
                xy[i] = 0
        if xy[2] > self.boundary[0]:
            xy[2] = self.boundary[0]
        if xy[3] > self.boundary[1]:
            xy[3] = self.boundary[1]
        return self.idx, self.cls, xy, self.conf[0], self.vector


class Boxes:
    # 存储结果，全是torch
    def __init__(self, result, chose):
        self.xy = result[0][chose]
        self.conf = result[1][chose]
        self.cls = result[2][chose]

    def is_in(self, xy_range, expand=0):  # 返回在xy_range里所有的box
        idx = []
        for i in range(self.xy.shape[0]):
            xy = self.xy[i]
            if xy[0] > xy_range[0] - expand and xy[1] > xy_range[1] - expand and xy[2] < xy_range[2] + expand and \
                    xy[3] < xy_range[3] + expand:
                idx.append(i)
        return idx, self.xy[idx], self.conf[idx]

    def is_like(self, xy_center, utc):  # 返回中心点近似于xy_center的box
        idx = []
        utc = utc / 2
        c_range = [xy_center[0] - utc, xy_center[1] - utc, xy_center[0] + utc, xy_center[1] + utc]
        for i in range(self.xy.shape[0]):
            xc = (self.xy[i][2] - self.xy[i][0]) / 2
            yc = (self.xy[i][3] - self.xy[i][1]) / 2
            if c_range[0] < xc < c_range[2] and c_range[1] < yc < c_range[3]:
                idx.append(i)
        return idx, self.xy[idx], self.conf[idx]

    def drop(self, idx):  # 去掉idx位置的数据
        if type(idx) is int:
            idx = [idx]
        chose = []
        for i in range(len(self.xy)):
            if i not in idx:
                chose.append(i)
        self.xy = self.xy[chose]
        self.conf = self.conf[chose]
        self.cls = self.cls[chose]

    def b_append(self, boxes_in, nms_id):
        self.xy = torch.cat((self.xy, boxes_in.xy[nms_id]), 0)
        self.conf = torch.cat((self.conf, boxes_in.conf[nms_id]), 0)
        self.cls = torch.cat((self.cls, boxes_in.conf[nms_id]), 0)
        pass

    def distribute(self, xys_pre, utc):
        # 计算相似度
        res = [[] for _ in range(len(xys_pre))]  # 返回值

        if self.xy.shape[0] == 0:
            return res
        sims = []
        for xy in xys_pre:
            temp = [xy for _ in range(self.xy.shape[0])]
            temp = torch.Tensor(temp) - self.xy
            sim = torch.norm(temp, dim=1)
            # sim = torch.nn.functional.cosine_similarity(self.xy, temp)
            sims.append(sim.numpy())
        sims = np.array(sims)  # (3, 30)
        if sims.shape[0] == 0:
            return res

        idx_in = []  # 各个pre里包含的框id
        for i in range(len(xys_pre)):
            idx, _, _ = self.is_in(xys_pre[i], expand=utc[i])
            idx_in.append(idx)
        for i in range(sims.shape[1]):
            min_idx = int(np.argmin(sims[:, i]))  # 从属最小的pre的id
            if sims[min_idx][i] < utc[min_idx]:  # 足够相似
                res[min_idx].append(i)
            else:
                for j in range(len(xys_pre)):
                    if i in idx_in[j]:  # 不够相似，但在某个pre内
                        res[j].append(i)
        return res

    def b_nms(self, idx, conf):
        nms_id = torchvision.ops.nms(self.xy[idx], self.conf[idx], conf)[0]
        return self.xy[idx[nms_id]], self.conf[idx[nms_id]], self.cls[idx[nms_id]]


def listdir_abs(folder):
    # 类似于os.listdir，返回每个文件的绝对路径
    files = os.listdir(folder)
    for i in range(len(files)):
        files[i] = folder + '\\' + files[i]
    return files
