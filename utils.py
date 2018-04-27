
Data_Prefix='/home/dzf/2016'
DATA_SET_NAME='2016'
data_subset = ['train', 'train-aug', 'validation', 'validation-aug', 'heatmap']

TRAIN_TUMOR_WSI_PATH = Data_Prefix+'/Train_Tumor'
TRAIN_NORMAL_WSI_PATH = Data_Prefix+'/Train_Normal'
TRAIN_TUMOR_MASK_PATH = Data_Prefix+'/Ground_Truth/Mask'

EVAL_DIR=Data_Prefix+'/Testset'
EVAL_LOGS=Data_Prefix+'/eval-logs'



EXTRACTED_PATCHES_NORMAL_PATH = Data_Prefix+'/Extracted_Negative_Patches/'
EXTRACTED_PATCHES_POSITIVE_PATH = Data_Prefix+'/Extracted_Positive_Patches/'
EXTRACTED_PATCHES_MASK_POSITIVE_PATH = Data_Prefix+'/Extracted-Mask-tumor/'
PROCESSED_PATCHES_TUMOR_NEGATIVE_PATH = Data_Prefix+'/Extracted_Negative_but_Tumor_Patches/'

TRAIN_TF_RECORDS_DIR = Data_Prefix+'/tf-records/'
PREFIX_SHARD_VALIDATiION = 'validation'
PATCHES_VALIDATION_DIR = Data_Prefix+'/Validation-Set/'

#about tf-records
PREFIX_SHARD_TRAIN = 'train'
PATCHES_TRAIN_DIR = Data_Prefix

#TRAIN_DIR=RuningPath+'/img_datasets/Processed/training/model8/'
#FINE_TUNE_MODEL_CKPT_PATH=RuningPath+'/img_datasets/Processed/training/model5/'

N_TRAIN_SAMPLES = 288000
N_VALIDATION_SAMPLES = 10000
N_SAMPLES_PER_TRAIN_SHARD = 1000
N_SAMPLES_PER_VALIDATION_SHARD = 250

NUM_POSITIVE_PATCHES_FROM_EACH_BBOX=500
NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX=500

PATCH_SIZE = 256


PATCH_INDEX_NEGATIVE = 700000
PATCH_INDEX_POSITIVE = 700000
PATCH_NORMAL_PREFIX = 'normal_'
PATCH_TUMOR_PREFIX = 'tumor_'

#augment after 1st train and refine
PATCHES_VALIDATION_AUG_NEGATIVE_PATH=Data_Prefix+'/Patches-validation-aug-negative/'
PATCHES_VALIDATION_AUG_POSITIVE_PATH=Data_Prefix+'/Patches-validation-aug-positive/'
PATCH_AUG_NORMAL_PREFIX = 'aug_false_normal_'
PATCH_AUG_TUMOR_PREFIX = 'aug_false_tumor_'

TRAIN_DIR=Data_Prefix+'/events_logs/'
FINE_TUNE_MODEL_CKPT_PATH=Data_Prefix+'/Fine-tune-models/'
EVAL_MODEL_CKPT_PATH=Data_Prefix+'/Eval-models/'
TRAIN_MODELS=Data_Prefix+'/Train-models/'
TRAIN_LOGS=Data_Prefix+'/train-logs/'

def get_filename_from_path(file_path):
    path_tokens = file_path.split('/')
    filename = path_tokens[path_tokens.__len__() - 1].split('.')[0]
    return filename