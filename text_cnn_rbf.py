import tensorflow as tf
import numpy as np
from sklearn import svm

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, svm_c, l2_reg_lambda=0.0, trainable=True):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x") # 为输入的数据创建占位符，在任何阶段都可使用它向模型输入数据。第二个参数是输入张量的形状，none表示该维度可以为任意值。而在我们模型中该维度表示批处理大小默认为64。
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y") # 标记好的输出的分类
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob") # 保存数据用的，第一个参数是数据类型第二个参数是数据结构

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0) # l2正则项的loss 初始化为常数0
        #设立了一个常量记录L2正则损失，每当出现新的变量时就会用变量的L2正则损失乘上L2正则损失权值加入到这个l2_loss里面来。

        # Embedding layer
        # 这一层的作用是将词汇索引映射到低维度的词向量进行表示。它本质是一个我们从数据中学习得到的词汇向量表。
        with tf.device('/cpu:0'), tf.name_scope("embedding"): # name_scope创建一个新的名称范围，用于TenosorBoard
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), # 随机初始化参数w，也就是词向量的矩阵，最大句子长度*词向量维度
                trainable=trainable, name="W")
            # 把随机初始化的句子中的数字看做索引，寻找对应的词向量
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x) # 选取词向量矩阵里索引对应的元素，input_x为索引
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)# TensorFlow 的卷积操作 conv2d 需要一个四维的输入数据，对应的维度分别是批处理大小，宽度，高度和通道数。我们需要手动添加通道数
        #定义了词嵌入矩阵，将输入的词id转化成词向量，这里的词嵌入矩阵是可以训练的，最后将词向量结果增加了一个维度，为了匹配CNN的输入

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size): # 为每种大小的filter创建一个卷积层，他们产生不同维度的张量，最后将这些张量合并为一个大的特征向量
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters] # 卷积核的维度
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W") # 这里的w就是指不同的卷积核,是由正太分布随机产生数值，产生的数与均值的差不超过两倍的标准方差
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b") # 生成维度为num_filters的bias常量，数值为0.1
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded, # 卷积层的输入是四维向量
                    W, # W是filter的参数,self.W才是输入的词向量
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu") # 对经过卷积得到的输出结果进行非线性处理之后的结果
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1], # 卷积执行后得到的向量的shape，sequence length是句子长度
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled) # 得到的张量维度是 [batch_size, 1, 1, num_filters]。这实质上就是一个特征向量，其中最后一个维度就是对应于我们的特征。

        with tf.name_scope("get_features"):
            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)  # 3表示在第三维上进行连接，也就是在num_filters上进行向量的连接
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total],
                                          name="features")  # 尽可能展平 num_filters_total是总的特征数

            # W_fc1 = tf.Variable(tf.truncated_normal([num_filters_total, 128], stddev=0.1), name="W")
            # b_fc1 = tf.Variable(tf.constant(0.1, shape=[128]))
            # self.h_fc1 = tf.nn.relu(tf.nn.xw_plus_b(self.h_pool_flat, W_fc1, b_fc1))

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob) # 一般设0.5的比例
            print("h_drop:", self.h_drop)
        # # Final (unnormalized) scores and predictions
        # with tf.name_scope("output"):
        #     W = tf.get_variable( # 全连接层分类器要更新的参数，用于将之前提取到的特征进行综合
        #         "W",
        #         shape=[num_filters_total, num_classes],
        #         initializer=tf.contrib.layers.xavier_initializer())
        #     b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
        #     l2_loss += tf.nn.l2_loss(W) # 加上正则项
        #     l2_loss += tf.nn.l2_loss(b)
        #     self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
        #     self.predictions = tf.argmax(self.scores, 1, name="predictions")
        #
        # # Calculate mean cross-entropy loss
        # with tf.name_scope("loss"):
        #     losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y) # 定义损失函数，用标记的真正的y值减去训练出来的
        #     self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        #
        # # Accuracy
        # # 一个batch更新一次参数后的准确率
        # with tf.name_scope("accuracy"):
        #     correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))  # 比较两个tensor的值一样就返回true
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        #     # cast是做一个映射把bool投成float类型然后求均值

        # SVM classifier
        with tf.name_scope("output"):
            W = tf.get_variable(  # 全连接层分类器要更新的参数
                        "W",
                        shape=[num_filters_total, 1],
                        initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
            # # Gaussian (RBF) kernel
            # # 该核函数用矩阵操作来表示
            # # 在sq_dists中应用广播加法和减法操作
            # # 线性核函数可以表示为：my_kernel=tf.matmul（x_data，tf.transpose（x_data）)
            # gamma = tf.constant(-10.0)
            # dist = tf.reduce_sum(tf.square(tf.transpose(self.h_drop)), 1)
            # print(dist)
            # dist = tf.reshape(dist, [-1, 1])
            # sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(self.h_drop, tf.transpose(self.h_drop)))),
            #                   tf.transpose(dist))
            # print("sq_dists:", sq_dists)
            # my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))
            # print("my_kernel:", my_kernel)
            # self.scores= tf.nn.xw_plus_b(my_kernel, W, b, name="scores")
            # print("self.scores:", self.scores)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            print("self.scores:", self.scores)
            predictions = tf.sign(self.scores) # 定义模型输出。利用符号函数实现，结果大于0时，输出1，小于0时，输出-1。
            self.predictions = tf.reshape(predictions, [-1, ], name="predictions")
            print("self.predictions:", self.predictions)

        # Calculate mean Hinge loss
        with tf.name_scope("loss"):
            l2_loss += tf.nn.l2_loss(W)  # 加上正则项
            l2_norm = tf.reduce_sum(tf.square(W))
            # self.loss = tf.reduce_mean(tf.maximum(0., 1. - self.scores * self.input_y)) + l2_reg_lambda * l2_loss
            self.loss = tf.reduce_mean(tf.maximum(0., 1. - self.scores * self.input_y)) + svm_c * l2_norm
            print("self.loss:", self.loss)

        # Accuracy
        # 一个batch更新一次参数后的准确率
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(predictions, self.input_y) # 比较两个tensor的值一样就返回true
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            # cast是做一个映射把bool投成float类型然后求均值
