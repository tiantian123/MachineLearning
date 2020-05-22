#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Tian Chen
# @times: 2020/5/19  9:52
# @File: TextClassify.py
# @email: chentianfighting@126.com
import os
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


def load_stop_words(file):
    with open(file, 'r', encoding='utf-8-sig') as f:
        stop_words = f.read().split('\n')
        # stop_words = stop_words.encode('utf-8').decode('utf-8-sig') # 列表头部\ufeff处理
        return stop_words


def load_data(basePath):
    word_label = []
    word_list = []
    for root, dirs, files in os.walk(basePath):
        for file in files:
            label = root.split('\\')[-1]
            word_label.append(label)
            with open(root+'\\'+file, 'r', encoding='gb18030') as f:
                text = list(jieba.cut(f.read()))
                word_list.append(' '.join(text))
    return word_list, word_label


def trainModel(trainData, trainLabel, testData, testLabel, stop_words):
    tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)
    train_feature = tf.fit_transform(trainData)
    test_feature = tf.transform(testData)

    clf = MultinomialNB(alpha=0.001).fit(train_feature, trainLabel)
    predicted_label = clf.predict(test_feature)

    x = metrics.accuracy_score(testLabel, predicted_label)

    return x


if __name__ == '__main__':
    stop = load_stop_words('./stop/stopword.txt')
    train_lst, train_label = load_data('G:\\git_project\\MachineLearning\\NaiveBayes\\train')
    test_lst, test_label = load_data('G:\\git_project\\MachineLearning\\NaiveBayes\\test')
    accuracy = trainModel(train_lst, train_label, test_lst, test_label, stop)
    print(accuracy)
