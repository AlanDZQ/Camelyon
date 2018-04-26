from __future__ import absolute_import #绝对引入
from __future__ import division #精确除法
from __future__ import print_function

import re
import tensorflow as tf
from inception_v3.slim import slim ##引入slim模块里面定义的v3模型

# 执行main函数之前首先进行flags的解析，
# 也就是说TensorFlow通过设置flags来传递tf.app.run()所需要的参数，
# 我们可以直接在程序运行前初始化flags，也可以在运行程序的时候设置命令行参数来达到传参的目的。
FLAGS = tf.app.flags.FLAGS

# 用多个GPU训练模型的时候，每个GPU上面都会对应一个tower，所以需要tower_name这个值来区分操作的对象 （通过前缀）
TOWER_NAME = 'tower'

# BN的移动平均值的衰减系数
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

# 移动平均值的衰减系数
MOVING_AVERAGE_DECAY = 0.9999

# 下面的方法是定义用于训练的inference模型
def inference(images, num_classes, for_training=False, restore_logits=True,
              scope=None):
    """
    :param images: 来自inputs()函数或者distorted_inputs()函数的返回值
    :param num_classes: 类别的数量
    :param for_training: 表明这个inference模型是用于训练的
    :param restore_logits: 对用过不同的类别数量来微调模型是有帮助的
    :param scope: 识别在哪一个tower上
    :return: 三个返回值
    """
    # 与BN有关的参数列表，其中decay用到了上面定义的好的参数值
    # epsilon是用来避免出现0变化
    batch_norm_params = {
        'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
        'epsilon': 0.001
    }

    # slim模块的ops里面定义了一些经典的网络层，比如卷积层、全连接层
    # 下面这行代码将卷积层和全连接层的权重衰减系数都设置成0.00004
    with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
        # 下面这行代码是将卷积函数里面的标准差设置成0.1，激活函数选用relu，batch_norm的参数用上面定义好的参数
        with slim.arg_scope([slim.ops.conv2d],
                            stddev=0.1,
                            activation=tf.nn.relu,
                            batch_norm_params = batch_norm_params):
            # 下面的操作是将一些参数传到slim模块原装的V3模型里面，具体点就是
            # inception_v3()这个函数，这里参数有一些是上面定义好的或者inference()
            # 函数传进来的，也有现在才定义的
            logits, end_points = slim.inception.inception_v3(
                images,
                dropout_keep_prob=0.8,
                num_classes=num_classes,
                is_training=for_training,
                restore_logits=restore_logits,
                scope=scope
            )

            # inception_v3()的输出值end_points作为参数值，
            # 调用下面定义好的方法来实现训练过程的可视化
            _activation_summaries(end_points)

            # 辅助分类节点，帮助预测分类结果
            auxiliary_logits = end_points['aux_logits']

            # 返回值：logits输出层的输出，未归一化的概率 inception_v3()的返回值之一
            #        auxiliary_logits
            #        end_points['predictions']
            return logits, auxiliary_logits, end_points['predictions']


# 为上面的inference()方法服务, 训练过程可视化
def _activation_summary(x):
    """
    用直方图的形式或者通过对激活函数稀疏度的度量来总结激活函数
    :param x: Tensor
    :return:
    """
    # 用正则表达式模块(re)的sub()函数来处理TOWER_NAME
    tensor_name = re.sub('%s_[0-9]*/]' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

# 用到上面定义好的_activation_summary()函数，只不过这里的参数不是tensor，而是endpoints
# 解析endpoints里面的tensor，然后调用上面定义好的方法，实现训练过程的可视化
def _activation_summaries(endpoints):
    with tf.name_scope('summaries'):
        for act in endpoints.values():
            _activation_summary(act)


# 将模型里面所有的损失加到一起
def loss(logits, labels, batch_size=None):
    """
    需要注意的是最后的损失并没有返回，而是放到slim.losses里面，损失在tower_loss()里面进行叠加，
    最后叠加的结果就是我们想要的总体损失
    :param logits: 来自上面定义的inference()函数的返回值，一个列表，列表里面的每一个条目都是一个二维的浮点张量tensor
    :param labels: 来自distorted_inputs()函数或者inputs()函数的返回值，上面定义的inference()函数也用到这两个
                   函数的返回值images，是shape [batch_size]的一维张量tensor
    :param batch_size:
    :return:
    """
    # batch_size的预设值是None，如果真的没有传入batch_size这个参数，那么就用flags里面已经定义好的batch_size值
    if not batch_size:
        batch_size = FLAGS.batch_size

    # 将标签labels重新塑造成一个密集的张量，为slim模块losses类计算总体损失服务
    sparse_labels = tf.reshape(labels, [batch_size, 1])
    # 指数 indices
    indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
    concated = tf.concat_v2([indices, sparse_labels],1)
    num_classes = logits[0].get_shape()[-1].value
    # 稀疏转密集
    dense_labels = tf.sparse_to_dense(concated,
                                      [batch_size, num_classes],
                                      1.0, 0.0)

    # 开始使用slim模块的losses类计算损失
    # main softmax prediction选用的损失函数是交叉熵损失，标签参数选用的是上面定义好的密集标签
    slim.losses.cross_entropy_loss(logits[0],
                                   dense_labels,
                                   label_smoothing=0.1,
                                   weight=1.0)

    # auxiliary softmax head选用的损失函数也是交叉熵损失，这里用到一个scope参数来确保它在辅助模型域里面
    slim.losses.cross_entropy_loss(logits[1],
                                   dense_labels,
                                   label_smoothing=0.1,
                                   weight=0.4,
                                   scope='aux_loss')
