#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Tian Chen
# @times: 2020/3/31  15:47
# @File: Breast_svm.py
# @email: chentianfighting@126.com
# 数据分析加载
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 数据集划分
from sklearn.model_selection import train_test_split, StratifiedKFold
# 数据预处理
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
# 模型加载
from sklearn import svm
# 模型结果评估
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# load the data
data = pd.read_csv('breast_data.csv')

# data analysis
pd.set_option('display.max_columns', None)
print(data.columns)
# print(data.head(5))
# print(data.describe())
# 数据预处理
feature_mean = list(data.columns[2:12])
feature_se = list(data.columns[12:22])
feature_worst = list(data.columns[22:32])
data.drop("id", axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
label = data['diagnosis']

# 数据探索
sns.countplot(data['diagnosis'], label="Count")
plt.show()
corr = data[feature_mean].corr()
plt.figure(figsize=(14, 14))
sns.heatmap(corr, annot=True)
plt.show()

# 数据集划分
train_x, test_x, train_y, test_y = train_test_split(data[feature_mean], label, test_size=0.3, shuffle=True)
model_svm = svm.SVC()

# 数据归一化,构建模型pipeline
pipe_svm = Pipeline([
    ('sc', StandardScaler()),
    ('pca', PCA(n_components=6)),
    ('power_trans', PowerTransformer()),
    ('svm', model_svm)
])

pipe_svm.fit(train_x, train_y)
y_predict = pipe_svm.predict(test_x)

print('准确率', accuracy_score(y_predict, test_y))
# print('绝对值偏差均值', mean_absolute_error(y_predict, test_y))
# print('二乘偏差均值', mean_squared_error(y_predict, test_y))

# 采用K折交叉验证对模型进行训练和预测
k = 5
kf = StratifiedKFold(n_splits=k, shuffle=True)
acc = []
precision = []
recall = []
auc = []
for train_index, test_index in kf.split(data[feature_mean], label):
    x_train, x_test = data[feature_mean].loc[train_index], data[feature_mean].loc[test_index]
    y_train, y_test = label.loc[train_index], label[test_index]
    pipe_svm.fit(x_train, y_train)
    y_predict = pipe_svm.predict(x_test)
    k_acc = accuracy_score(y_test, y_predict)
    print(f"accuracy score is: {k_acc}")
    acc.append(k_acc)
    k_precision = precision_score(y_test, y_predict)
    print(f"precision score is: {k_precision}")
    precision.append(k_precision)
    k_recall = recall_score(y_test, y_predict)
    print(f"recall score is: {k_recall}")
    recall.append(k_recall)
    k_auc = roc_auc_score(y_test, y_predict)
    print(f"auc: {k_auc}")
    auc.append(k_auc)
    print("--"*30)

print(f'average accuracy score is: {np.mean(acc)}')
print(f"average precision score is: {np.mean(precision)}")
print(f"average recall score is: {np.mean(recall)}")
print(f"average auc value is: {np.mean(auc)}")