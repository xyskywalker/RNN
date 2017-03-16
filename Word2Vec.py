# coding:utf-8
# 实现 Word2Vec

import collections
import math
import os
import random
import zipfile
import numpy as np
import tensorflow as tf

# 数据文件: http://www.mattmahoney.net/dc/text8.zip
filename = 'downloads/text8.zip'
# 取频次最高的5000词汇
vocabulary_size = 50000

# 读取数据
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


# 构建数据集
def build_dataset(words):
    # Top 50000 之外的单词计数
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

words = read_data(filename)
print('Data size:                %d' % len(words))
data, count, dictionary, reverse_dictionary = build_dataset(words)
del words
print('Most common words (+UNK): %s' % count[:10])
print('Sample data:              %s' % data[:10])
print('                          %s' % [reverse_dictionary[i] for i in data[:10]])




