# !/usr/local/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/1/1 21:00
# @Author  : Aurope
# @FileName: lgbTrain.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import lightgbm as lgb
import joblib

basePath = '/Users/aurope/PythonProjects/BigData-NLP'

train_df = pd.read_csv(basePath + '/dataset/train_set.csv', sep='\t')
test_df = pd.read_csv(basePath + '/dataset/test_a.csv', sep='\t')

train_df['text_split'] = train_df['text'].apply(lambda x: str(x.split()))
test_df['text_split'] = test_df['text'].apply(lambda x: str(x.split()))

# 词向量
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
cv_scores = []  # 每一轮线下的验证成绩

for i, (train_index, eval_index) in enumerate(kf.split(train_term_doc)):
    print(len(train_index), len(eval_index))

    # 训练集
    X_train = train_term_doc[train_index]
    y_train = train_df['label'][train_index]

    # 验证集
    X_eval = train_term_doc[eval_index]
    y_eval = train_df['label'][eval_index]

    model = lgb.LGBMClassifier(boosting_type='gbdt',
                               num_leaves=2 ** 5,
                               max_depth=-1,
                               learning_rate=0.15,
                               n_estimators=2000,
                               objective='multiclass',
                               subsample=0.7,
                               reg_lambda=10,
                               n_jobs=16,
                               num_class=19,
                               silent=True,
                               random_state=2019,
                               colsample_bylevel=0.5,
                               min_child_weight=1.5,
                               metric='multi_logloss'
                               )
    model.fit(X_train, y_train, eval_set=(X_eval, y_eval), early_stopping_rounds=20)

    # 对于验证集进行预测
    eval_prob = model.predict_proba(X_eval)
    train_matrix[eval_index] = eval_prob.reshape((X_eval.shape[0], 14))  # array

    eval_pred = np.argmax(eval_prob, axis=1)
    score = f1Score(y_eval, eval_pred)
    cv_scores.append(score)

    print("Current score is ", score)

    # 保存第N此FOLD模型结果
    with open(file='../modelLgb/modelScore.txt', mode='a+') as f:
        print(f'model {i}:', file=f)
        print("Current score is ", score, file=f)
    f.close()

    # 保存模型
    joblib.dump(model, f'./lgbSavedModel/model-{i}-lgb.pkl')
    # 对于测试集进行预测
    test_prob = model.predict_proba(test_term_doc)
    test_pre_matrix[i, :, :] = test_prob.reshape((test_term_doc.shape[0], 14))

all_pred = np.argmax(train_matrix, axis=1)
score = f1Score(train_df['label'], all_pred)

print("Final score is ", score)

test_pred = test_pre_matrix.mean(axis=0)
test_pred = np.argmax(test_pred, axis=1)
test_df['label'] = test_pred
test_df['label'].to_csv("./sortResult/sortResult_lgb.csv", index=False)
