import numpy as np
import os
import random
import shutil

# 将文件夹下存于images和labels文件夹里all文件夹中的数据分割开，存到下属train和val文件夹中（并清空原有文件）
dataset_folder = r'F:\dataset'
setup_num = 1  # 用作文件后缀，区分各个分割方式或数据集,=1时，创建的train文件夹名为train01

list_images = os.listdir(dataset_folder + r'\images\all')  # 获取所有图片文件名
list_labels = os.listdir(dataset_folder + r'\labels\all')  # 获取所有label文件名


def main():
    # ---检查文件缺失
    names = []  # 读取所有文件名
    for item in list_images:
        names.append(item[:item.rfind('.')])
    for item in list_labels:
        if not item[:item.rfind('.')] in names:  # 逐一检查
            print('Label', item, 'is missing')

    id_train, id_val = rand(len(list_images))  # 按0.2的比例选取val

    # ---根据id将文件转存到对应文件夹，删除原有文件
    check_folder(dataset_folder + r'\images\train%02d' % setup_num)
    check_folder(dataset_folder + r'\images\val%02d' % setup_num)
    check_folder(dataset_folder + r'\labels\train%02d' % setup_num)
    check_folder(dataset_folder + r'\labels\val%02d' % setup_num)

    #  复制文件
    for i in id_train:
        shutil.copyfile(dataset_folder + r'\images\all' + '\\' + list_images[i],
                        dataset_folder + r'\images\train%02d' % setup_num + '\\' + list_images[i])
        shutil.copyfile(dataset_folder + r'\labels\all' + '\\' + list_labels[i],
                        dataset_folder + r'\labels\train%02d' % setup_num + '\\' + list_labels[i])
    for i in id_val:
        shutil.copyfile(dataset_folder + r'\images\all' + '\\' + list_images[i],
                        dataset_folder + r'\images\val%02d' % setup_num + '\\' + list_images[i])
        shutil.copyfile(dataset_folder + r'\labels\all' + '\\' + list_labels[i],
                        dataset_folder + r'\labels\val%02d' % setup_num + '\\' + list_labels[i])


def check_folder(f_name):
    #  若无文件夹，创建新文件夹，若有则清空文件
    if not os.path.exists(f_name):
        os.makedirs(f_name)
    else:
        for f in os.listdir(f_name):
            os.remove(os.path.join(f_name, f))


def rand(n, ratio=None):
    #  根据ratio把n范围内的整数分为两组
    if ratio is None:
        ratio = [0.9, 0.1]
    id1 = random.sample(range(n), int(ratio[1]*n))  # 抽取的图形
    id0 = []
    for i in range(n):
        if i not in id1:
            id0.append(i)
    return id0, id1


if __name__ == '__main__':
    main()


