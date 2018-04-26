from __future__ import absolute_import #绝对引入
from __future__ import division #引入精确除法
from __future__ import print_function

from abc import ABCMeta #Python本身不提供抽象类和接口机制，要想实现抽象类，可以借助abc模块。
from abc import abstractmethod
import os

import tensorflow as tf

#引入路径管理类
import external.utils as utils

#训练的数据集合
PROCESSED_PATCHES_TRAIN = '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/data/CAMELYON16/Processed/' \
                                   'patch-based-classification/raw-data/train/'
PROCESSED_PATCHES_TRAIN_NEGATIVE = PROCESSED_PATCHES_TRAIN + 'label-0/'
PROCESSED_PATCHES_TRAIN_POSITIVE = PROCESSED_PATCHES_TRAIN + 'label-1/'

#验证的数据集合
PROCESSED_PATCHES_VALIDATION = '/home/millpc/Documents/Arjun/Study/Thesis/CAMELYON16/data/CAMELYON16/' \
                                        'Processed/patch-based-classification/raw-data/validation/'
PROCESSED_PATCHES_VALIDATION_NEGATIVE = PROCESSED_PATCHES_VALIDATION + 'label-0/'
PROCESSED_PATCHES_VALIDATION_POSITIVE = PROCESSED_PATCHES_VALIDATION + 'label-1/'


FLAGS = tf.app.flags.FLAGS #通过设置flags来传递tf.app.run()所需要的参数

# Basic model parameters. tf_record数据目录
tf.app.flags.DEFINE_string('data_dir', utils.TRAIN_TF_RECORDS_DIR,
                           """Path to the processed data, i.e. """
                           """TFRecord of Example protos.""")


class Dataset(object):
    """A simple class for handling data sets.
        数据集合的管理类
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, subset, tf_records_dir=None, num_patches=0):
        """Initialize dataset using a subset and the path to the data.
            通过子集和数据的路径来初始化数据集合
        """
        assert subset in self.available_subsets(), self.available_subsets() #断言是声明其布尔值必须为真的判定
        self.name = name
        self.subset = subset
        self.heatmap_tf_records_dir = tf_records_dir
        self.heatmap_num_patches = num_patches

    def is_heatmap_data(self):
        return self.subset == 'heatmap'

    def num_classes(self):
        """Returns the number of classes in the data set.
            返回数据集的类目数
        """
        return 2  # [0, 1]

    def num_examples_per_epoch(self):
        """Returns the number of examples in the data subset.
           返回数据集子集的所有示例数
        """
        if self.subset == 'train':
            return utils.N_TRAIN_SAMPLES
        elif self.subset == 'validation':
            return utils.N_VALIDATION_SAMPLES
        else:  # hear-map
            return self.heatmap_num_patches

    @abstractmethod
    def download_message(self):
        """Prints a download message for the Dataset."""
        pass

    def num_examples_per_shard(self):
        """Returns the number of examples in one shard.
           返回一个分片里面的示例数
        """
        if self.subset == 'train':
            return utils.N_SAMPLES_PER_TRAIN_SHARD
        elif self.subset == 'validation':
            return utils.N_SAMPLES_PER_VALIDATION_SHARD
        else:  # hear-map
            return self.heatmap_num_patches

    def available_subsets(self):
        """Returns the list of available subsets."""
        return utils.data_subset

    def data_files(self):
        """Returns a python list of all (sharded) data subset files.

        Returns:
          python list of all (sharded) data set files.
        Raises:
          ValueError: if there are not data_files matching the subset.
        """
        tf_record_pattern = os.path.join(FLAGS.data_dir, '%s-*' % self.subset)
        data_files = tf.gfile.Glob(tf_record_pattern)
        print(data_files)
        if not data_files:
            print('No files found for dataset %s/%s at %s' % (self.name,
                                                              self.subset,
                                                              FLAGS.data_dir))

            self.download_message()
            exit(-1)
        return data_files

    def data_files_heatmap(self):
        """Returns a python list of all (sharded) data subset files.

        以python列表的形式返回分片里面的所有数据子集文件
        Returns:
          python list of all (sharded) data set files.
        Raises:
          ValueError: if there are not data_files matching the subset.
        """
        assert self.heatmap_tf_records_dir is not None

        tf_record_pattern = os.path.join(self.heatmap_tf_records_dir, '%s-*' % self.subset)
        data_files = tf.gfile.Glob(tf_record_pattern)
        if not data_files:
            print('No files found for dataset %s/%s at %s' % (self.name,
                                                              self.subset,
                                                              self.heatmap_tf_records_dir))

            self.download_message()
            exit(-1)
        return data_files

    def reader(self):
        """Return a reader for a single entry from the data set.
        返回针对数据集里面单个条目的读取
        See io_ops.py for details of Reader class.

        Returns:
          Reader object that reads the data set.
        """
        return tf.TFRecordReader()
