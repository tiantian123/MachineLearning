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
    """
    基本路径下所有数据的加载
    :param basePath: 数据基本路径
    :return: 分词后的文档列表和标签列表
    """
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
    """
    多项式贝叶斯分类器 模型训练及预测
    :param trainData: 训练集数据
    :param trainLabel: 训练集标签
    :param testData: 测试集数据
    :param testLabel: 测试集标签
    :param stop_words: 停用词列表
    :return: 模型的准确率
    """
    tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)
    # fit_transform 拟合模型，返回文本矩阵
    train_feature = tf.fit_transform(trainData)
    # 用trainData fit 过了，测试数据只需要进行transform
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
