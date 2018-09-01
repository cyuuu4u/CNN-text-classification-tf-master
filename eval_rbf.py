#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn_rbf import TextCNN
from tensorflow.contrib import learn
import csv
from sklearn import svm

# Parameters
# 命令行参数
# ==================================================

# Training Data
tf.flags.DEFINE_string("positive_train_file", "./data/rt_train/rt-polarity_train.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_train_file", "./data/rt_train/rt-polarity_train.neg", "Data source for the negative data.")
# tf.flags.DEFINE_string("positive_train_file", "./data/imdb/pos_train.txt", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_train_file", "./data/imdb/neg_train.txt", "Data source for the negative data.")

# Testing Data
tf.flags.DEFINE_string("positive_test_file", "./data/rt_test/rt-polarity_test.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_test_file", "./data/rt_test/rt-polarity_test.neg", "Data source for the negative data.")
# tf.flags.DEFINE_string("positive_test_file", "./imdb_test/pos.txt", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_test_file", "./imdb_test/neg.txt", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)") # 一个batch的默认大小
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# SVM parameters
tf.flags.DEFINE_float("gamma", 0.01, "svm parameter")
tf.flags.DEFINE_float("svm_c", 0.1, "svm parameter")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

print("Loading training data of svm...")
x_text, y_labels = data_helpers.load_data_and_labels(FLAGS.positive_train_file, FLAGS.negative_train_file) # 返回[x_text,y]

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train: # 测试所有输入的训练样本
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_test_file, FLAGS.negative_test_file)
    y_test = np.argmax(y_test, axis=1)  # 返回每句数据标签中值为1的下标值，返回1是positive，返回0是negative
    y_test = np.array([-1 if y == 0 else 1 for y in y_test])

else: # 测试自己输入的样本
    x_raw = []
    stopword = ''
    str = ''
    label = ''
    print("Enter your English texts, end with empty line:")
    for line in iter(input, stopword):
        str += line + '\n'
    x_raw = str.split(sep="\n")
    del x_raw[-1]
    print("Enter corresponding polarity, 1 means positive, -1 means negative end with empty line:")
    for line in iter(input, stopword):
        label += line + '\n'
    if label != '':
        y_test = label.split(sep="\n")
        del y_test[-1]
        y_test = [int(n) for n in y_test]
    else:
        y_test = None
    # x_raw = ["a masterpiece four years in the making", "everything is off."]
    # y_test = [1, -1]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x = np.array(list(vocab_processor.fit_transform(x_text)))
y_train = np.argmax(y_labels, axis=1)
y = np.array([-1 if y == 0 else 1 for y in y_train])

x_test = np.array(list(vocab_processor.transform(x_raw)))

# shuffled
np.random.seed(10)  # seed用于指定随机数生成
shuffle_indices = np.random.permutation(np.arange(len(y)))# arange(len(y))返回一个array([0,1,2...,10661]) shuffle_indiced返回一个array([7767,...10433])随机序列
x_train = x[shuffle_indices] # 打乱了数据 还是返回每个句子为一行的随机数矩阵后面补零
y_train = y[shuffle_indices] # 生成对应的label的矩阵每行为一个句子，是[1,0]或[0,1]
train_labels = y_train.reshape(-1,)
print("\nEvaluating...\n")

# Evaluation
# ==================================================
# 找到最新的时间点应用模型
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    # 打开会话,准备传入构造好的graph,交给后台运算
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        features = graph.get_operation_by_name("get_features/features").outputs[0]

        # 以上分别为保存,输出 输入占位符 的设置

        train_features = sess.run(features, {input_x: x_train, dropout_keep_prob: 1.0})
        test_features = sess.run(features, {input_x: x_test, dropout_keep_prob: 1.0})
        # # Generate batches for one epoch
        # # 取batches,每FLAGS.batch_size个list(x_test),作一个batch
        # batches_train = data_helpers.batch_iter(list(x_train), FLAGS.batch_size, 1, shuffle=False)
        # batches_test = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # # Collect the predictions here
        # # 判断输出,用来和标签比对,即知道正确率,见下
        # train_features = []
        # test_features = []
        #
        # for x_train_batch in batches_train:
        #     batch_train_features = sess.run(features, {input_x: x_train_batch, dropout_keep_prob: 1.0})
        #     print(batch_train_features)
        #     train_features = np.concatenate((train_features, batch_train_features), axis=0)
        # for x_test_batch in batches_test:
        #     batch_test_features = sess.run(features, {input_x: x_test_batch, dropout_keep_prob: 1.0})
        #     test_features = np.concatenate((test_features, batch_test_features), axis=0)
        #     # 循环相连接,把所有的x_test_batch跑出来的结果x_test_batch,首尾相连

# # Print accuracy if y_test is defined
# if y_test is not None:
#     correct_predictions = float(sum(all_predictions == y_test)) # 正确预测的个数
#     print("Total number of test examples: {}".format(len(y_test)))
#     print("Accuracy: {:g}".format(correct_predictions/float(len(y_test)))) # 预测的正确率
#     # y_test=所有的标签,correct_predictions=正确率求和,两者求商,为正确率
#
# # Save the evaluation to a csv
# # 准备数据,存入该路径下的prediction.csv文件
# predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
# out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
# print("Saving evaluation to {0}".format(out_path))
# with open(out_path, 'w') as f:
#     csv.writer(f).writerows(predictions_human_readable)

# train svm
clf = svm.SVC(kernel='rbf', gamma=FLAGS.gamma, C=FLAGS.svm_c)
clf.fit(train_features, train_labels)
all_predictions = clf.predict(test_features)
# Print accuracy if y_test is defined
if y_test is not None:
    TruePositive = 0
    FalsePositive = 0
    TrueNegative = 0
    FalseNegative = 0
    flag = []
    for i in range(len(y_test)):
        if y_test[i] == 1 and all_predictions[i] == 1:
            TruePositive += 1
            flag.append("right")
        if y_test[i] == 1 and all_predictions[i] == -1:
            FalsePositive += 1
            flag.append("wrong")
        if y_test[i] == -1 and all_predictions[i] == -1:
            TrueNegative += 1
            flag.append("right")
        if y_test[i] == -1 and all_predictions[i] == 1:
            FalseNegative += 1
            flag.append("wrong")
    correct_predictions = float(sum(all_predictions == y_test)) # 正确预测的个数
    precision_pos = float(TruePositive / (TruePositive + FalsePositive))
    precision_neg = float(TrueNegative / (TrueNegative + FalseNegative))
    recall_pos = float(TruePositive / (TruePositive + FalseNegative))
    recall_neg = float(TrueNegative / (TrueNegative + FalsePositive))
    F1_pos = float((precision_pos * recall_pos * 2) / (precision_pos + recall_pos))
    F1_neg = float((precision_neg * recall_neg * 2) / (precision_neg + recall_neg))
    test_acc = correct_predictions / float(len(y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Test Accuracy: {:g}".format(test_acc))
    print("Precision_pos: {:g}, Recall_pos: {:g}, F1_pos: {:g}".format(precision_pos, recall_pos, F1_pos))
    print("Precision_neg: {:g}, Recall_neg: {:g}, F1_neg: {:g}".format(precision_neg, recall_neg, F1_neg))
    print("")
    # y_test=所有的标签,correct_predictions=正确率求和,两者求商,为正确率
    predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions, flag))
else:
    predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))


# Save the evaluation to a csv
# 准备数据,存入该路径下的prediction.csv文件
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)