# coding:utf-8
# 实现 Word2Vec

import collections
import math
import random
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 数据文件: http://www.mattmahoney.net/dc/text8.zip
filename = 'downloads/text8.zip'
# 取频次最高的5000词汇
vocabulary_size = 50000


# 读取数据
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


# 可视化函数
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    plt.savefig(filename)


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
    # 输出结果：
    # data: 编码，每个词按词频排序的位置(从1开始，0是UNK)
    # count: 词频统计表(第一个是UNK和他的计数，之后是top 5000的词和对应的计数)
    # dictionary: 词汇表,Key:词，Value:词频统计表中的位置
    # reverse_dictionary: 反转的词汇表,Key:词频统计表中的位置，Value:词
    return data, count, dictionary, reverse_dictionary

words = read_data(filename)
print('Data size:                %d' % len(words))
data, count, dictionary, reverse_dictionary = build_dataset(words)
del words
print('Most common words (+UNK): %s' % count[:11])
print('Sample data:              %s' % data[:10])
print('                          %s' % [reverse_dictionary[i] for i in data[:10]])

# 定义全局变量
data_index = 0


# 返回训练的batch数据
# batch_size: 每一个batch的尺寸
# num_skips: 针对每个单词生成多少样本
# skip_window: 单词最远能联系的距离，1代表只能跟紧邻他的两个单词生成样本
# num_skips不能大于skip_window的两倍，并且batch_size必须是num_skips的整数倍
def generate_batch(batch_size, num_skips, skip_window):
    # 声明为global变量
    global data_index
    # 添加断言，验证参数的准确性
    assert batch_size%num_skips == 0
    assert num_skips <= 2*skip_window
    # 初始化batch和labels数组
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # 代表对某个单词创建样本时会使用到的单词数量，包括目标单词本身和前后的单词
    span = 2*skip_window + 1
    # 创建一个双向队列，最大容量为span，所以对他append变量时只会保留最后插入的span个变量
    buffer = collections.deque(maxlen=span)

    # 创建初始值
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    for i in range(batch_size//num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i*num_skips + j] = buffer[skip_window]
            labels[i*num_skips + j, 0] = buffer[target]

        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return batch, labels

'''
# 获取并打印范例数据
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print('%d:%s -> %d:%s' % (batch[i],
                              reverse_dictionary[batch[i]],
                              labels[i, 0],
                              reverse_dictionary[labels[i, 0]]
                              )
          )
'''

# 定义参数
batch_size = 128
# 单词转换为稠密向量的维度
embedding_size = 128
skip_window = 1
num_skips = 2

# 验证用数据参数
valid_size = 16
# 验证单词只抽取词频前100的单词
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
# 噪声单词数量
num_sampled = 64

# 创建默认图
graph = tf.Graph()
with graph.as_default():

    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/cpu:0'):
        # 随机生成所有单词的词向量
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        # 在词向量中寻找输入的对应向量
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        # 权重
        nce_weights = tf.Variable(
            tf.truncated_normal(
                [vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)
            )
        )
        # 偏移量
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # 损失函数
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=vocabulary_size
            )
        )
        # 优化器
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        norm = tf.sqrt(
            tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True)
        )
        normalized_embeddings = tf.div(embeddings, norm)
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

        init = tf.global_variables_initializer()

        num_steps = 100001
        with tf.Session(graph=graph) as session:
            session.run(init)
            print('Initialized')

            average_loss = 0
            for step in range(num_steps):
                batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 2000 == 0:
                    print('Average loss as step %d : %d' % (step, average_loss))
                    average_loss = 0

                if step % 10000 == 0:
                    sim = similarity.eval()
                    for i in range(valid_size):
                        valid_word = reverse_dictionary[valid_examples[i]]
                        top_k = 8
                        nearest = (-sim[i, :]).argsort()[1:top_k+1]
                        log_str = 'Nearest to %s:' % valid_word
                        for k in range(top_k):
                            close_word = reverse_dictionary[nearest[k]]
                            log_str = '%s %s,' % (log_str, close_word)
                        print(log_str)

            final_embeddings = normalized_embeddings.eval()

            # 结果可视化
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            plot_only = 500
            low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
            labels = [reverse_dictionary[i] for i in range(plot_only)]
            plot_with_labels(low_dim_embs, labels)




