#!/usr/bin/env python
# encoding: utf-8
"""
@author: wushaohong
@time: 2020-05-13 10:37
"""
from collections import defaultdict


class HMM:
    def __init__(self):
        self.status = ['B', 'M', 'E', 'S']
        self.pi = {}  # 初始状态概率
        self.array_A = {}  # 状态转移概率矩阵
        self.array_B = {}  # 发射概率矩阵
        self.dict = {}
        self.init_array()
        self.words = set()
        self.average_p = 0.0

    def init_array(self):
        """
        初始化矩阵
        :return:
        """
        for s in self.status:
            self.pi[s] = 0
            self.dict[s] = defaultdict(int)
            self.array_B[s] = {}
            self.array_A[s] = {}
            for s2 in self.status:
                self.array_A[s][s2] = 0

    def get_array_B(self, words):
        """
        根据词表，获取发射概率矩阵
        :param words:
        :return:
        """

        def get_tag(word):
            if not word:
                return
            length = len(word)
            if length == 1:
                self.dict['S'][word] += 1
            else:
                num_M = length - 2
                self.dict['B'][word[0]] += 1
                for i in range(num_M):
                    self.dict['M'][word[i + 1]] += 1
                self.dict['E'][word[-1]] += 1

        def create_array(status):
            d = self.dict[status]
            count = sum(d.values())
            for key, value in d.items():
                self.array_B[status][key] = value / count

        list(map(get_tag, words))
        list(map(create_array, self.status))

    def get_array_A(self, docs):
        """
        通过训练数据，获取初始状态概率分布和状态转移概率矩阵
        :param docs:
        :return:
        """

        def calculate(doc):
            doc = doc.strip().split()
            if not doc:
                return
            string = ''
            for word in doc:
                length = len(word)
                if length == 1:
                    string += 'S'
                else:
                    num_M = length - 2
                    string += 'B' + num_M * 'M' + 'E'
            self.pi[string[0]] += 1
            for i in range(1, len(string)):
                self.array_A[string[i - 1]][string[i]] += 1

        list(map(calculate, docs))
        count_pi = sum(self.pi.values())
        for s in self.status:
            self.pi[s] = self.pi[s] / count_pi
            count = sum(self.array_A[s].values())
            for s2 in self.status:
                self.array_A[s][s2] = self.array_A[s][s2] / count
        self.words = set(''.join(docs))
        self.average_p = 1 / len(self.words)

    def train(self, words, docs):
        self.get_array_A(docs)
        for s in self.status:
            self.array_B[s] = {word: 0 for word in self.words}
        self.get_array_B(words)

    def viterbi(self, doc):
        """
        viterbi算法
        :param doc:
        :return:
        """
        tags = ['B', 'M', 'E', 'S']
        length = len(tags)
        probabilities = [self.pi[tag] * self.array_B[tag].get(doc[0], self.average_p) for tag in tags]
        temp_p = [0] * length
        temp_tags = [''] * length
        for i in range(1, len(doc)):
            for k, s in enumerate(self.status):
                ps = [probabilities[j] * self.array_A[tags[j][-1]][s] * self.array_B[s].get(doc[i], self.average_p) for
                      j in range(length)]
                max_p = max(ps)
                index = ps.index(max_p)
                temp_p[k] = max_p
                temp_tags[k] = tags[index] + s
            probabilities = list(temp_p)
            tags = list(temp_tags)
        max_p = max(probabilities)
        index = probabilities.index(max_p)
        tag = tags[index]
        text = ''
        for i, t in enumerate(tag):
            text += doc[i]
            if t == 'S' or t == 'E':
                text += ' '
        return text

    def test_many(self, docs, save2output=True):
        res = []
        for doc in docs:
            text = self.viterbi(doc)
            res.append(text)
        if save2output:
            with open('output.txt', 'w')as f:
                f.write('\n'.join(res))

    def test_one(self, doc):
        print('After HMM seg:', self.viterbi(doc))

    def main(self, words, docs, test_docs):
        self.train(words, docs)
        self.test_many(test_docs)


if __name__ == '__main__':
    words = []
    train_docs = []
    with open('./CTB_training_words.utf8') as f:
        words = f.read()
        words = words.split('\n')
    with open('./CTBtrainingset.txt') as f:
        train_docs = f.read()
        train_docs = train_docs.split('\n')

    test_doc = '但是后来呢就是我们的记者可能摄影记者放松警惕了。'
    hmm = HMM()
    hmm.train(words, train_docs)
    hmm.test_one(test_doc)
    test_docs = []
    with open('./CTBtestingset.txt') as f:
        test_docs = f.read()
        test_docs = test_docs.split('\n')
    hmm.test_many(test_docs)
