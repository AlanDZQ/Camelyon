"""
在单个GPU上面评估inception
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, '/home/arjun/MS/Thesis/CAMELYON-16/source')
import math
import os.path
import time
from datetime import datetime

from inception_v3 import image_processing
from inception_v3 import  inception_model as inception_model
import external.utils as utils
import numpy as np
import sklearn as sk
import tensorflow as tf

from inception_v3.dataset import Dataset
from tensorflow.contrib import metrics

FLAGS = tf.app.flags.FLAGS

# checkpoint的保存路径
CKPT_PATH = utils.EVAL_MODEL_CKPT_PATH

# 数据集合的名称
DATA_SET_NAME = 'TF-Records'

# 写日志的目录
tf.app.flags.DEFINE_string('eval_dir', utils.EVAL_DIR,
                           """Directory where to write event logs.""")

# 读取模型检查点的目录
tf.app.flags.DEFINE_string('checkpoint_dir', utils.TRAIN_DIR,
                           """Directory where to read model checkpoints.""")

# 执行评估模型操作的频率
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
# 是否就执行一次评估
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")

# 需要执行评估操作的数据实例的数量
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.
                            We have 10000 examples.""")

# subset 这次选用的是'validation'
tf.app.flags.DEFINE_string('subset', 'validation',
                           """Either 'validation' or 'train'.""")

BATCH_SIZE = 100

def _eval_once(saver, summary_writer, accuracy, summary_op, confusion_matrix_op):
    """
    执行一次评估
    :param saver: 保存器
    :param summary_writer:
    :param accuracy:
    :param summary_op:
    :param confusion_matrix_op:
    :return:
    """
    with tf.Session() as sess:
        # 打印一下检查点路径
        print(FLAGS.checkpoint_dir)
        ckpt = None
        if CKPT_PATH is not None:
            saver.restore(sess, CKPT_PATH)
            # 从检查点的路径里面得到全局步数的信息
            global_step = CKPT_PATH.split('/')[-1].split('-')[-1]
            print('Succesfully loaded model from %s at step=%s.' %
                  (CKPT_PATH, global_step))

        # ckpt的预设值就是None
        elif ckpt is None:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            # 如果给定的检查点目录里面有检查点的信息，那么就将模型的检查点目录打印出来
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                # 如果模型的检查点路径是绝对路径,就将该路径放到保存器里面
                if os.path.isabs(ckpt.model_checkpoint_path):
                    saver.restore(sess,ckpt.model_checkpoint_path)
                else:
                    # 不是绝对路径就把检查点的相对路径放到保存器里面
                    saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
                                                     ckpt.model_checkpoint_path))

                # 假设模型检查点路径是如下形式，从中提取出全局步数global_step
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print('Succesfully loaded model from %s at step=%s.' %
                      (ckpt.model_checkpoint_path, global_step))
        else:
            print('No checkpoint file found')
            return

        # 开始了序列执行
        coord = tf.train.Coordinator() # 创建的协调器，用来管理线程
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / BATCH_SIZE))

            # 计算预测正确的个数
            total_correct_count = 0
            total_false_positive_count = 0
            total_false_negative_count = 0
            total_sample_count = num_iter * BATCH_SIZE
            step = 0

            print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
            start_time = time.time()
            while step < num_iter and not coord.should_stop():
                correct_count, confusion_matrix = \
                    sess.run([accuracy, confusion_matrix_op])

                total_correct_count += np.sum(correct_count)
                total_false_positive_count += confusion_matrix[0][1]
                total_false_negative_count += confusion_matrix[1][0]

                print('correct_count(step=%d): %d / %d' % (step, total_correct_count, BATCH_SIZE * (step + 1)))
                print('\nconfusion_matrix:')
                print(confusion_matrix)
                print('total_false_positive_count: %d' % total_false_positive_count)
                print('total_false_negative_count: %d' % total_false_negative_count)

                step += 1
                if step % 20 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / 20.0
                    examples_per_sec = BATCH_SIZE / sec_per_batch
                    print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                          'sec/batch)' % (datetime.now(), step, num_iter,
                                          examples_per_sec, sec_per_batch))
                    start_time = time.time()

            # 计算准确率
            precision = total_correct_count / total_sample_count
            print('%s: precision = %.4f [%d examples]' %
                  (datetime.now(), precision, total_sample_count))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision', simple_value=precision)
            summary_writer.add_summary(summary, global_step)

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

# 参数dense_labels： 密集标签
# logits ：未归一化的概率， 运行sklearn的相关方法计算评估的一些指标
def calc_metrics(dense_labels, logits):
    print("Precision", sk.metrics.precision_score(dense_labels, logits))
    print("Recall", sk.metrics.recall_score(dense_labels, logits))
    print("f1_score", sk.metrics.f1_score(dense_labels, logits))
    print("confusion_matrix")
    print(sk.metrics.confusion_matrix(dense_labels, logits))

# 在数据集合上评估模型的具体步骤
def evaluate(dataset):
    with tf.Graph().as_default():
        # 从数据集合里面获取images和labels
        images, labels = image_processing.inputs(dataset, BATCH_SIZE)

        # 给数据集合标签集合里面的类别数量+1
        # 这里的+1是用标签0来标识未使用的背景类
        num_classes = dataset.num_classes()

        # 建立一个Graph，这个图的作用就是从inference模型里面计算未归一化的概率（预测）
        logits, _, _ = inception_model.inference(images,num_classes)

        # 把稀疏标签密集化
        sparse_labels = tf.reshape(labels, [BATCH_SIZE, 1])
        indices = tf.reshape(tf.range(BATCH_SIZE), [BATCH_SIZE, 1])
        concated = tf.concat(1, [indices, sparse_labels])
        num_classes = logits[0].get_shape()[-1].value
        dense_labels = tf.sparse_to_dense(concated,
                                          [BATCH_SIZE, num_classes],
                                          1, 0)
        confusion_matrix_op = metrics.confusion_matrix(labels, tf.argmax(logits, axis=1))

        # 计算预测值, 输出最接近的前几名
        accuracy = tf.nn.in_top_k(logits, labels, 1)

        # 恢复用于评估移动平均版本的学习变量
        variable_averages = tf.train.ExponentialMovingAverage(
            inception_model.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # 基于TF的Summaries集合建立summary操作
        summary_op = tf.summary.merge_all()

        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, graph_def=graph_def)

        # 这里调用了上面定义好的一次评估的方法
        while True:
            _eval_once(saver, summary_writer, accuracy, summary_op, confusion_matrix_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)

# 获取用于评估的数据集合，执行评估操作
dataset = Dataset(DATA_SET_NAME, utils.data_subset[2])
evaluate(dataset)



