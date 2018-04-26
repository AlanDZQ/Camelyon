"""
用多个同步更新的GPU来训练inception_model
"""
from __future__ import absolute_import # 绝对引入
from __future__ import division # 精确除法
from __future__ import print_function

import sys
# #新添加的目录会优先于其他目录被import检查
sys.path.insert(0, '/home/arjun/MS/Thesis/CAMELYON-16/source')

import copy #Python中的对象之间赋值时是按引用传递的，如果需要拷贝对象，需要使用标准库中的copy模块。
import os.path
import re

import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from inception_v3 import image_processing # 导入预处理模块
from inception_v3 import inception_model as inception_model #导入inception_model
from inception_v3.dataset import Dataset # 数据集管理类
from inception_v3.slim import slim # slim模块
import  external.utils as utils # 路径管理类

# 数据集合的名称
DATA_SET_NAME = 'Camelyon'

# 设置flags来传递tf.app.run()所需要的参数
FLAGS = tf.app.flags.FLAGS

# 写日志和检查点的目录
# checkpoint: 依赖代码而创建模型的一种格式
tf.app.flags.DEFINE_string('train_dir', utils.TRAIN_DIR,
                           """Directory where to write event logs """
                           """and checkpoint.""")

# 要运行的batches数目，最大 1M
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")

# subset的值要么是'train' 要么是 'validation'
# 这里首先选择的是'train'
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")

# 使用的GPU数量 2
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")

# 是否记录设备的放置情况: 否
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# 微调是否设置，如果设置了微调，那么就随机初始化最后一层的权重以便于在新的任务上训练网络：没有设置
tf.app.flags.DEFINE_boolean('fine_tune', False,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")

# 预训练模型的检查点路径，如果指定了相关路径，那么就在开始任何训练之前重新恢复预训练的模型
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', utils.FINE_TUNE_MODEL_CKPT_PATH,
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")

# 下面是有关学习率的相关参数设置，至于学习率该如何控制，参考以下论文链接：
# http://arxiv.org/abs/1404.5997

# 最开始的学习率
tf.app.flags.DEFINE_float('initial_learning_rate', 0.01,  # 1*e-2
                          """Initial learning rate.""")

# 每次学习率衰退之后的Epochs数量
tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                          """Epochs after which learning rate decays.""")

# 每次学习率衰退之后程序执行的步数
tf.app.flags.DEFINE_integer('num_steps_per_decay', 60000,
                            """Steps after which learning rate decays.""")
# 学习率的衰退因子
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                          """Learning rate decay factor.""")

#与学习率调度有关的常量
RMSPROP_DECAY = 0.9
RMSPROP_MOMENTUM = 0.9
RMSPROP_EPSILON = 1.0

# 在运行模型的时候，这个方法计算单个tower上的总体损失，这里有一个'batch splitting'的
# 关机制，这意味着如果batch size的数量是32，而gpu的数量是2，那么每个tower上面将会运行batch size = 16的图像
def _tower_loss(images, labels, num_classes, scope, reuse_variables=None):
    '''

    :param images: 四维张量tensor  [batch_size, FLAGS.image_size, FLAGS.image_size, 3]
    :param labels: 一维的整数张量
    :param num_classes: 种类的数量
    :param scope: 模型运行在哪个tower上的标识
    :param reuse_variables:
    :return: 返回一个包含一批数据总体损失的张量
    '''

    # logits: 未归一化的概率， 输出层的输出，一般也就是 softmax层的输入
    # 微调模型的时候，我们不会恢复logits, 这个参数值设置是为inception_model里面的inference()方法服务的
    restore_logits = not FLAGS.fine_tune

    # 之前在inception_model这个文件里面写一个inference模型，也就是将一些参数放到
    # slim模块中设计好的模型里面
    # 这里我们建立inference的graph, reuse的值用方法里面传入的参数值
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        # 开始用到了inception_model里面定义的inference方法，来训练模型
        # 基本上都是用到_tower_loss() 方法里面传入的参数
        logits = inception_model.inference(images,
                                           num_classes,
                                           for_training=True,
                                           restore_logits = restore_logits,
                                           scope= scope)

        # 借用inception_model里面定义好的loss函数来计算tower上的总体损失
        # 这里就是计算每个tower上面的batch数目
        split_batch_size = images.get_shape().as_list()[0]
        # logit参数就是inference()方法的输出值之一， labels是传进来的参数值
        inception_model.loss(logits,labels,batch_size=split_batch_size)

        # 将当前tower的损失组织起来, scope来指定是哪个tower
        losses = tf.get_collection(slim.losses.LOSSES_COLLECTION, scope)

        # 将这些损失组织好了之后，计算这些损失的总和，和的结果就是当前tower的总体损失
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

        # 计算个体损失和总体损失的移动平均 moving average
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        loss_averages_op = loss_averages.apply(losses+ [total_loss])

        # 将个体损失和总体损失标量化
        for l in losses + [total_loss]:
            # 如果是一个多GPU训练会话的话，那么就通过正则表达式移除相关的前缀名，这样有助于可视化
            loss_name = re.sub('%s_[0-9]*/' % inception_model.TOWER_NAME, '', l.op.name)

            # 将每个损失的名称改为'(raw)'， 此时损失原本的名称表示的是移动平均化之后的损失
            tf.summary.scalar(loss_name + ' (raw)', l)
            tf.summary.scalar(loss_name, loss_averages.average(1))

        with tf.control_dependencies([loss_averages_op]):
            total_loss = tf.identity(total_loss)
        return total_loss #返回总体损失、

# 计算所有tower里面每个共享变量的平均梯度，这个功能提供了跨所有tower的同步点
def _average_gradients(tower_grads):
    '''
     输入参数是 （梯度，变量）的元组列表，
     返回值还是（梯度，变量）的元组列表，只不过所有tower的梯度都被平均化过
    :param tower_grads:
    :return:
    '''
    average_grads = []
    # zip函数将可迭代对象处理成一个元组列表: ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            # 添加0维度到梯度里面来展示tower的信息
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        # 在 tower 维度0 进行平均化操作
        grad = tf.concat_v2(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        #变量是冗余的，因为它们在所有tower里面共享的，所以我们只需要返回第一个指向变量的tower指针
        v = grad_and_vars[0][1]
        grad_and_vars = (grad, v) # 经过处理之后的（梯度、变量）元组
        average_grads.append(grad_and_vars)
    return average_grads

# 在一个训练集合上通过一些步骤来进行训练
def train(dataset):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # 创建一个变量来计算train的调用次数，这个数字也是batch的数量和gpu数量的乘积
        global_step = tf.get_variable(
            'global_step',[],
            initializer=tf.constant_initializer(0), trainable=False)

        # 计算学习率的调度
        num_batches_per_epoch = (dataset.num_examples_per_epoch() /
                                 FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

        # 学习率会根据步数以指数方式进行衰减
        # 通常的操作是首先使用较大学习率(目的：为快速得到一个比较优的解)
        # 然后通过迭代逐步减小学习率(目的：为使模型在训练后期更加稳定)
        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                        global_step,
                                        60000,
                                        FLAGS.learning_rate_decay_factor,
                                        staircase=True)

        # 创建一个梯度下降的优化器
        opt = tf.train.GradientDescentOptimizer(lr)

        # 获取图像和标签，然后根据GPU的个数，将它们分到各自的batch里面
        assert FLAGS.batch_size % FLAGS.num_gpus == 0, (
            'Batch size must be divisible by number of GPUs')
        split_batch_size = int(FLAGS.batch_size / FLAGS.num_gpus)

        # 当GPU tower的数量增加的时候，预处理的线程也会相应的增加
        num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus
        # 这里就用到预处理的distorted_inputs()方法增加训练的样本
        images, labels = image_processing.distorted_inputs(
            dataset,
            num_preprocess_threads=num_preprocess_threads)

        # 这里用到copy模块的copy()方法
        input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

        num_classes = dataset.num_classes()

        # 依据tower将images和labels划开
        images_splits = tf.split(images, FLAGS.num_gpus, axis=0)
        labels_splits = tf.split(labels, FLAGS.num_gpus, axis=0)

        # 计算每一个tower里面模型的梯度
        tower_grads = []
        reuse_variables = None
        for i in range(FLAGS.num_gpus):
            # 这里选定的设备是GPU
            with tf.device('/gpu:%d' % i):
                # 定义scope参数
                with tf.name_scope('%s_%d' % (inception_model.TOWER_NAME, i)) as scope:
                    # 强制所有变量停留在CPU上面, 用到slim模块的参数作用域来处理
                    with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
                        # 计算模型里面一个tower的损失，这里用到的就是上面定义的损失函数
                        loss = _tower_loss(images_splits[i],
                                           labels_splits[i],
                                           num_classes,
                                           scope,
                                           reuse_variables)
                    # 注意之前225行的reuse_variables的参数值设置的None
                    reuse_variables = True

                    # 保留最后一个tower的summaries, scope还是上面定义的scope
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # 仅仅保留最后的tower的BN更新操作
                    # 理想情况下，我们应该从所有tower获取更新，但是这些统计数据积累速度非常之快
                    # 所以我们可以忽略来自其他tower的统计数据，而不会造成重大损害
                    # 用到了slim模块定义的相关方法
                    batch_norm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION,
                                                           scope)

                    #计算tower上的那一个batch数据的梯度, loss是上面算出来的指定tower的总体损失
                    grads = opt.compute_gradients(loss)

                    # 跟踪所有tower上的梯度
                    tower_grads.append(grads)

        # 这里用到了上面定义好的方法来计算每个梯度上的平均值
        grads = _average_gradients(tower_grads)

        # 为输入处理和总体步数，添加一个summarize
        summaries.extend(input_summaries)

        # 添加一个跟踪学习率的summary
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # 为梯度添加直方图
        for grad, var in grads:
            if grad is not None:
                summaries.append(
                    tf.summary.histogram(var.op.name + '/gradients', grad))

        # 应用相关梯度来调整共享变量， 选用优化器的相关方法
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # 为训练变量添加直方图
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # 跟踪训练变量的移动平均
        # 需要注意的是我们维护了BN总体数据的一个双重平均，这是为了向后兼容我们以前的模型
        variable_averages = tf.train.ExponentialMovingAverage(
            inception_model.MOVING_AVERAGE_DECAY, global_step)

        # “双重平均” 另外一平均
        variable_to_average = (tf.trainable_variables() + tf.moving_average_variables())
        variable_averages_op = variable_averages.apply(variable_to_average)

        # 将所有的更新操作组织到一个单独的训练操作里面
        batch_norm_updates_op = tf.group(*batch_norm_updates)
        train_op = tf.group(apply_gradient_op, variable_averages_op, batch_norm_updates_op)

        #创建一个保存器
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1000)

        # 从最后一个tower的summarize里面建立起summary操作
        summary_op = tf.summary.merge(summaries)

        # 初始化
        init = tf.initialize_all_variables()

        # 开始在Graph上面运行，allow_soft_placement必须设置为True
        # 这是为了在GPU上面把tower建立起来，因为有一些操作没有GPU的实现
        config = tf.ConfigProto(allow_soft_placement = True, log_device_placement=FLAGS.log_device_placement)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        sess = tf.Session(config=config)
        sess.run(init) # 运行

        if FLAGS.pretrained_model_checkpoint_path:
            # 断言这个为真： tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
            print('model path: %s' % FLAGS.pretrained_model_checkpoint_path)
            variables_to_restore = tf.get_collection(
                slim.variables.VARIABLES_TO_RESTORE)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
            print('%s: Pre-trained model restored from %s' %
                  (datetime.now(), FLAGS.pretrained_model_checkpoint_path))

        # 开始序列化运行
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(
            FLAGS.train_dir,
            graph_def=sess.graph.as_graph_def(add_shapes=True))

        # 一些日志记录
        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, duration))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # 定期保存模型的检查点checkpoint
            if step % 5000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step) # 上面定义的保存器用在了这里

dataset = Dataset(DATA_SET_NAME, utils.data_subset[0]) # 把数据集合里面的数据放进来
train(dataset) # 执行训练操作








