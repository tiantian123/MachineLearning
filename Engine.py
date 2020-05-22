#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Tian Chen
# @times: 2020/4/2  11:18
# @File: Engine.py
# @email: chentianfighting@126.com


class SearchEngieBase(object):
    def __init__(self):
        pass

    def add_corpus(self, file_path):
        with open(file_path, 'r') as fin:
            text = fin.read()
        self.process_corpus(file_path, text)

    def process_corpus(self, id, text):
        raise Exception('process_corpus not implemented.')

    def search(self, query):
        raise Exception('search not implementd.')

class SimpleEngine(SearchEngieBase):
    def __init__(self):
        # super(SimpleEngine, self).__init__()
        super().__init__() # python3 直接调用父类的初始化函数
        self.__id_to_texts = {}

    def process_corpus(self, id, text):
        self.__id_to_texts[id] = text

    def search(self, query):
        result = []
        for id, text in self.__id_to_texts.items():
            if query in text:
                result.append(id)
        return result


def main(search_engine):
    for file_path in ['1.txt', '2.txt']:
        search_engine.add_corpus(file_path)

    while True:
        query = input()
        results = search_engine.search(query)
        print(f'found {len(results)} result(s)')
        for idx in results:
            print(results)


search_engine = SimpleEngine()
main(search_engine)
