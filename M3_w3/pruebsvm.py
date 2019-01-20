import os
import numpy as np
from PIL import Image
from scipy.misc import imresize

IMG_SIZE    = 32
BATCH_SIZE  = 16

pu=os.listdir('datasets/MIT_split/train/forest/')

print(pu)

images_train = []
labels_train = []
j = 0
for i in ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']:
    directory_train = 'datasets/MIT_split/train/' + i
    for image in os.listdir(directory_train):
        x = np.asarray(Image.open(os.path.join(directory_train, image)))
        #x = np.expand_dims(imresize(x, (IMG_SIZE, IMG_SIZE, 3)), axis=0)
        images_train.append(x)
        labels_train.append(j)
    j = j + 1