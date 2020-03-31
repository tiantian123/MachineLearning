#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Tian Chen
# @times: 2020/3/31  15:47
# @File: Breast_svm.py
# @email: chentianfighting@126.com
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# load data set
data = pd.read_csv('data.csv')

# Data exploration
