# coding=utf-8

__author__ = '陈杰'

import os, glob
from corpus import *
from lda import *
from operator import itemgetter
import sys

reload(sys)
sys.setdefaultencoding('utf-8')


def print_topic_word(lda_model, topk, filepath):
    print '写' + filepath + '......'
    V = len(lda_model.corpus.vocabulary)
    with open(filepath, "w") as f:
        for k in range(lda_model.number_of_topics):
            word_count = lda_model.topic_word_count[k, :]
            word_index_count = []
            for v in range(V):
                word_index_count.append([v, word_count[v]])
            word_index_count = sorted(word_index_count, key=itemgetter(1), reverse=True)
            f.write("Topic No:" + str(k) + ":\n")
            for i in range(topk):
                index = word_index_count[i][0]
                f.write(
                    lda_model.corpus.vocabulary[index] + '\t' + str(lda_model.topic_word_distribution[k, index]) + "\t")
            f.write("\n")


def print_document_topic(lda_model, topk, filepath):
    print '写' + filepath + '......'
    with open(filepath, "w") as f:
        M = len(lda_model.corpus.documents)
        for m in range(M):
            topic_count = lda_model.document_topic_count[m, :]
            topic_index_count = []
            for k in range(lda_model.number_of_topics):
                topic_index_count.append([k, topic_count[k]])
            topic_index_count = sorted(topic_index_count, key=itemgetter(1), reverse=True)
            f.write("Document No:" + str(m) + ":\n")
            for i in range(topk):
                index = topic_index_count[i][0]
                f.write("topic" + str(index) + '\t' + str(lda_model.document_topic_distribution[m, index]) + "\t")
            f.write("\n")


# 建立停用词库
def build_stop_words(stop_words_path):
    STOP_WORDS = {}
    with open(stop_words_path, 'r') as stop_words_file:
        for line in stop_words_file:
            line = line.strip()
            STOP_WORDS[line.decode('utf-8')] = 1
    return STOP_WORDS


# 20个主题,alpha2.5,beta0.1,迭代次数100
def main(number_of_topics=20, max_iter=100, alpha=2.5, beta=0.1):
    # 停用词
    STOP_WORDS = build_stop_words('stop_words.txt')

    # 建立语料
    news_corpus = corpus()

    # 新闻路径
    news_path = r'./news'
    for news in glob.glob(os.path.join(news_path, '*.txt')):
        documents = read_document(news, STOP_WORDS)
        for document in documents:
            news_corpus.add_document(document)

    # 建立词典
    news_corpus.build_vocabulary()

    # 建立LDA模型
    lda_model = lda(news_corpus, number_of_topics)

    # Gibbs Sampling
    lda_model.gibbs_sampling(max_iter, alpha, beta)

    # 写topic-word.txt
    print_topic_word(lda_model, 25, "./topic-word.txt")

    # 写document-topic.txt
    print_document_topic(lda_model, 10, "./document-topic.txt")


if __name__ == "__main__":
    main()