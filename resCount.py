# !/usr/local/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/1/2 18:08
# @Author  : Aurope
# @FileName: resCount.py
# @Software: PyCharm

import pandas as pd


def resCount(path):
    df = pd.read_csv(path, sep='\t')
    df = df['label'].value_counts()
    df.index = ['科技', '股票', '体育', '娱乐', '时政', '社会', '教育', '财经', '家居', '游戏', '房产', '时尚', '彩票', '星座']
    df.to_csv('./sortResult/resCount_lgb.csv', index=True, header=True, encoding='utf-8')
    print('结果统计完毕。')


if __name__ == '__main__':
    resCount('./sortResult/sortResult_lgb.csv')
    # resCount('./sortResult/sortResult_rnn.csv')
    print('Ok')