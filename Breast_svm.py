#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Tian Chen
# @times: 2020/3/31  15:47
# @File: Breast_svm.py
# @email: chentianfighting@126.com
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# load data set
data = pd.read_csv('breast_data.csv')

# Data exploration， set_option to show all columns
pd.set_option('display.max_columns',None)
# print(data.columns)
# print(data.head(5))
# print(data.describe())

# split feature into 3 group
features_mean = list(data.columns[2:12])
features_se = list(data.columns[12:22])
features_worst = list(data.columns[22:32])

# data clean
# del id
data.drop("id", axis=1, inplace=True)
# B is good, replace to 0; M is bad, replace to 1
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
"""
# show the result of diagnosis.
sns.countplot(data['diagnosis'], label="Count")
plt.show()
# using the heat map show the relationship among the feature_mean.
corr = data[features_mean].corr()
plt.figure(figsize=(16, 16))
# annot = true, show the data of every box.
sns.heatmap(corr, annot=True)
plt.show()
"""
# feature select
features_remain = ['radius_mean','texture_mean', 'smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean']
# extract 30% data as testing dataset, the others are training dataset
train, test = train_test_split(data, test_size=0.3)
train_X = train[features_worst]
train_y = train['diagnosis']
test_X = test[features_worst]
test_y = test['diagnosis']

# using Z-Score to normalization, ensure every feature data mean is 0, variance is 1
ss = StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.transform(test_X)

# Build SVM classification
model = svm.LinearSVC()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print("准确率: ", metrics.accuracy_score(prediction, test_y))

