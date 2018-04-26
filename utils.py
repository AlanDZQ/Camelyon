import os
import sys

#os.environ['Path'] = "D:\\program\\openslide-win64-20171122\\openslide-win64-20171122\\bin"+";"+os.environ['Path']
RuningPath = sys.path[0]

TRAIN_TUMOR_WSI_PATH = RuningPath+'/img_datasets/CAMELYON-16/Tumor'
TRAIN_NORMAL_WSI_PATH = RuningPath+'/img_datasets/CAMELYON-16/Normal'
TRAIN_TUMOR_MASK_PATH = RuningPath+'/img_datasets/CAMELYON-16/Mask'

PROCESSED_PATCHES_NORMAL_NEGATIVE_PATH =RuningPath+ '/img_datasets/normal-label-0/'
PROCESSED_PATCHES_TUMOR_NEGATIVE_PATH = RuningPath+'/img_datasets/tumor-label-0/'
PROCESSED_PATCHES_POSITIVE_PATH = RuningPath+'/img_datasets/label-1/'
PROCESSED_PATCHES_FROM_USE_MASK_POSITIVE_PATH = RuningPath+'/img_datasets/use-mask-label-1/'

PATCH_SIZE = 256
PATCH_NORMAL_PREFIX = 'normal_'
PATCH_TUMOR_PREFIX = 'tumor_'

TRAIN_TF_RECORDS_DIR = RuningPath+'/img_datasets/Processed/tf-records/'
PREFIX_SHARD_VALIDATION = 'validation'
PATCHES_VALIDATION_DIR = RuningPath+'/img_datasets/' + 'Processed/patch-based-classification/raw-data/validation/'

PATCHES_TRAIN_DIR =RuningPath+ '/img_datasets/' + 'train/'
PREFIX_SHARD_TRAIN = 'train'

TRAIN_DIR=RuningPath+'/img_datasets/Processed/training/model8/'
FINE_TUNE_MODEL_CKPT_PATH=RuningPath+'/img_datasets/Processed/training/model5/'

N_TRAIN_SAMPLES = 288000
N_VALIDATION_SAMPLES = 10000
N_SAMPLES_PER_TRAIN_SHARD = 1000
N_SAMPLES_PER_VALIDATION_SHARD = 250

