# !/usr/local/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/1/5 14:47
# @Author  : Aurope
# @FileName: visualization.py
# @Software: PyCharm

import csv
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

# 加载CSV文件


filePath1 = './sortResult/resCount_lgb.csv'
filePath2 = './sortResult/resCount_rnn.csv'


def getChineseFont():
    return FontProperties(fname='/System/Library/Fonts/PingFang.ttc')


with open(filePath1) as f1:
    # 加载CSV模块阅读器
    reader = csv.reader(f1)

    # 读取第一行表头
    header_row = next(reader)

    # 依次读取每行，并保存日期，最高温，最低温为列表
    name1, countNum1 = [], []
    for row in reader:
        cur_name = row[0]
        count = int(row[1])
        name1.append(cur_name)
        countNum1.append(count)
f1.close()

with open(filePath2) as f2:
    # 加载CSV模块阅读器
    reader = csv.reader(f2)

    # 读取第一行表头
    header_row = next(reader)

    # 依次读取每行，并保存日期，最高温，最低温为列表
    name2, countNum2 = [], []
    for row in reader:
        cur_name = row[0]
        count = int(row[1])
        name2.append(cur_name)
        countNum2.append(count)
f2.close()

# 图像分辨率，尺寸设置
fig = plt.figure(dpi=128, figsize=(10, 6))

# 标题设置
plt.title('分类结果', fontproperties=getChineseFont())

# X轴标签设置，自动更新格式
plt.xlabel('种类', fontproperties=getChineseFont())
fig.autofmt_xdate()

# Y轴标签和坐标范围设置
plt.ylabel('数量', fontproperties=getChineseFont())
plt.ylim(0, 10000)

# 刻度设置
plt.tick_params(axis='both', which='both', labelsize=8)

# 根据数据画折线图
plt.plot(name1, countNum1, label='LGB', marker="o", markersize=10, linewidth=3, c='red', alpha=0.5)
plt.plot(name2, countNum2, label='RNN', marker="X", markersize=10, linewidth=3, c='green', alpha=0.5)
# 区域填充
plt.fill_between(name1, countNum1, facecolor='blue', alpha=0.1)
plt.xticks(fontproperties=getChineseFont())
# 图像显示
plt.legend()
plt.grid()  # 添加网格
plt.show()
