# coding=utf-8

__author__ = '陈杰'

import jieba
import sys

reload(sys)
sys.setdefaultencoding('utf-8')


def read_document(document_path, STOP_WORDS):
    with open(document_path, 'r') as document_file:
        documents = []
        for line in document_file:
            document = []
            line = line.strip()
            words = list(jieba.cut(line))
            for i in words:
                try:
                    STOP_WORDS[i]
                except:
                    document.append(i)
            documents.append(document)
    return documents


class corpus(object):
    def __init__(self):
        self.documents = []
        self.vocabulary = []


    def add_document(self, document):
        self.documents.append(document)


    def build_vocabulary(self):
        words = set()
        for document in self.documents:
            for word in document:
                words.add(word)
        self.vocabulary = list(words)