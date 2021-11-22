from os.path import join
import torch
import pandas as pd

DATA_PATH = '/media/yangyuhui/DATA/sartorius/sartorius-cell-instance-segmentation'
SAMPLE_SUBMISSION = join(DATA_PATH, 'train')
TRAIN_CSV = join(DATA_PATH, 'train.csv')
TRAIN_PATH = join(DATA_PATH, 'train')
TEST_PATH = join(DATA_PATH, 'test')
device = torch.device("cuda:{}".format(0)) if torch.cuda.is_available() else torch.device('cpu')
df_train = pd.read_csv(TRAIN_CSV)
print(f'Training Set Shape: {df_train.shape} - {df_train["id"].nunique()} \
Images - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')