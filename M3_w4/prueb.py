
from sklearn.datasets import load_digits
from utils import *

IMG_SIZE = 30
BATCH_SIZE = 32
# num_epochs has to be an even number, because first iteration is training for num_epochs / 2
NUM_EPOCHS = 100
base_model_last_block_layer_number = 30
DATASET_DIR = 'MIT'
MODEL_FNAME = 'xception_finetuning.h5'
W_FNAME = 'WEIGHTS_xception_finetuning.h5'

directory_train = DATASET_DIR + '/train/'
train_data, labels_train = load_data(IMG_SIZE, directory_train)


