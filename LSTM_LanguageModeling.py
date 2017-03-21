# coding:utf-8
# LSTM实现语言模型

import time
import numpy as np
import tensorflow as tf
import reader


# 处理输入数据的class
class PTBInput(object):


    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size

