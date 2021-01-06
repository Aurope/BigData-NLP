# !/usr/local/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/1/2 20:35
# @Author  : Aurope
# @FileName: lgbPred.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import joblib

# 模型名称
n = 0

basePath = '/Users/aurope/PythonProjects/BigData-NLP'

# train_df = pd.read_csv(basePath + '/dataset/train_set.csv', sep='\t')
# test_df = pd.read_csv(basePath + '/dataset/test_a.csv', sep='\t')

train_df = pd.read_csv(basePath + '/dataset/train_set_tiny.csv', sep='\t')
test_df = pd.read_csv(basePath + '/dataset/test_a_tiny.csv', sep='\t')

train_df['text_split'] = train_df['text'].apply(lambda x: str(x.split()))
test_df['text_split'] = test_df['text'].apply(lambda x: str(x.split()))

word_vec = TfidfVectorizer(analyzer='word',
                           ngram_range=(1, 2),
                           min_df=3,
                           max_df=0.9,
                           use_idf=True,
                           max_features=3000,
                           smooth_idf=True,
                           sublinear_tf=True)
train_term_doc = word_vec.fit_transform(train_df['text_split'])
test_term_doc = word_vec.transform(test_df['text_split'])


def f1Score(y_true, y_pred):
    cur_score = f1_score(y_true, y_pred, average='macro')
    return cur_score


X_train, X_eval, y_train, y_eval = train_test_split(train_term_doc, train_df['label'], test_size=0.2, shuffle=True,
                                                    random_state=2019)  # split the training data

# CV
kf = KFold(n_splits=10, shuffle=True, random_state=666)
train_matrix = np.zeros((train_df.shape[0], 14))  # 记录验证集的概率

test_pre_matrix = np.zeros((10, test_df.shape[0], 14))  # 将5轮的测试概率分别保存起来



for i, (train_index, eval_index) in enumerate(kf.split(train_term_doc)):
    if i != n:
        continue

    print(len(train_index), len(eval_index))

    # 训练集
    X_train = train_term_doc[train_index]
    y_train = train_df['label'][train_index]

    # 验证集
    X_eval = train_term_doc[eval_index]
    y_eval = train_df['label'][eval_index]

    model = joblib.load(basePath + f'/modelLgb/model-{n}-lgb.pkl')

    # 对于验证集进行预测
    eval_prob = model.predict_proba(X_eval)
    train_matrix[eval_index] = eval_prob.reshape((X_eval.shape[0], 14))  # array

    eval_pred = np.argmax(eval_prob, axis=1)
    score = f1Score(y_eval, eval_pred)

    print("Current score is ", score)

    # 对于测试集进行预测
    test_prob = model.predict_proba(test_term_doc)
    test_pre_matrix[n, :, :] = test_prob.reshape((test_term_doc.shape[0], 14))

test_pred = test_pre_matrix.mean(axis=0)
test_pred = np.argmax(test_pred, axis=1)
test_df['label'] = test_pred
test_df['label'].to_csv("../sortResult/sortRes_tiny.csv", index=False, header=True, encoding='utf-8')
