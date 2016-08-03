# coding=utf-8

__author__ = '陈杰'

import numpy as np
from random import random
import sys

reload(sys)
sys.setdefaultencoding('utf-8')


def choose(pro):
    s = sum(pro)
    for i in range(len(pro)):
        pro[i] = pro[i] / s
    y = -1
    a = random()
    b = 0
    while a > 0:
        a = a - pro[b]
        y += 1
        b += 1
    return y


class lda(object):
    def __init__(self, corpus, number_of_topics):
        self.corpus = corpus
        self.number_of_topics = number_of_topics
        self.number_of_documents = len(corpus.documents)
        self.number_of_vocabulary = len(corpus.vocabulary)
        self.document_topic_count = np.zeros([self.number_of_documents, self.number_of_topics], dtype=np.int)
        self.document_topic_distribution = np.zeros([self.number_of_documents, self.number_of_topics], dtype=np.float)
        self.topic_word_count = np.zeros([self.number_of_topics, len(self.corpus.vocabulary)], dtype=np.int)
        self.topic_word_distribution = np.zeros([self.number_of_topics, len(self.corpus.vocabulary)], dtype=np.float)
        self.word_topic = []
        self.topic_count = np.zeros(self.number_of_topics)

    def gibbs_sampling(self, max_iter, alpha=0.1, beta=0.5):
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        print "初始化......"
        for document_index, document in enumerate(self.corpus.documents):
            word_topic = []
            for word in document:
                word_index = self.corpus.vocabulary.index(word)
                topic_index = np.random.randint(self.number_of_topics)
                word_topic.append(topic_index)
                self.document_topic_count[document_index, topic_index] += 1
                self.topic_word_count[topic_index, word_index] += 1
                self.topic_count[topic_index] += 1
            self.word_topic.append(np.array(word_topic))

        for i in range(self.max_iter):
            print "迭代第" + str(i + 1) + '......'
            for document_index, document in enumerate(self.corpus.documents):
                for word_where, word in enumerate(document):
                    word_index = self.corpus.vocabulary.index(word)
                    current_topic_index = self.word_topic[document_index][word_where]
                    self.document_topic_count[document_index, current_topic_index] -= 1
                    self.topic_word_count[current_topic_index, word_index] -= 1
                    self.topic_count[current_topic_index] -= 1

                    topic_distribution = (self.topic_word_count[:, word_index] + self.beta) * \
                                         (self.document_topic_count[document_index] + self.alpha) / \
                                         (self.topic_count + self.beta)
                    new_topic = choose(topic_distribution)
                    self.word_topic[document_index][word_where] = new_topic
                    self.document_topic_count[document_index, new_topic] += 1
                    self.topic_word_count[new_topic, word_index] += 1
                    self.topic_count[new_topic] += 1
        for document_index, document in enumerate(self.corpus.documents):
            a = (self.document_topic_count[document_index] + self.alpha) \
                / 1.0 / (sum(self.document_topic_count[document_index]) + self.alpha)
            self.document_topic_distribution[document_index] = (self.document_topic_count[document_index] + self.alpha) \
                                                               / 1.0 / (sum(
                self.document_topic_count[document_index]) + self.alpha)
            for word in document:
                word_index = self.corpus.vocabulary.index(word)
                self.topic_word_distribution[:, word_index] = (self.topic_word_count[:, word_index] + self.beta) \
                                                              / 1.0 / (self.topic_count + self.beta)