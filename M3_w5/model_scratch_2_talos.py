import os
import getpass

from utils import *
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape, Input, concatenate, GlobalAveragePooling2D, \
    MaxPooling2D, Conv2D, InputLayer,Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import datetime
from keras.layers.advanced_activations import *
from keras import regularizers
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras import backend as k
import talos as ta

# user defined variables
IMG_SIZE = 30
BATCH_SIZE = 32
NUM_EPOCHS = 80
DATASET_DIR = '/home/mcv/datasets/MIT_split'
MODEL_FNAME = 'MODEL_scratch.h5'
W_FNAME = 'WEIGHTS_MODEL_scratch.h5'


def create_Model(x_train, y_train, x_val, y_val, parameters):
    print('Building our scratch model...\n')

    model = Sequential()

    model.add(InputLayer(input_shape=[30, 30, 3], name='Input'))

    model.add(Conv2D(filters=parameters['filter_1'], kernel_size=parameters['kernel_size1'], strides=1, padding='same', activation=parameters['activation'], name='FC1'))
    model.add(MaxPooling2D(pool_size=5, padding='same'))

    model.add(Conv2D(filters=parameters['filter_2'], kernel_size=parameters['kernel_size2'], strides=1, padding='same', activation=parameters['activation'], name='FC2'))
    model.add(MaxPooling2D(pool_size=5, padding='same'))

    model.add(Conv2D(filters=parameters['filter_3'], kernel_size=parameters['kernel_size3'], strides=1, padding='same', activation=parameters['activation'], name='FC3'))
    model.add(MaxPooling2D(pool_size=5, padding='same'))

    model.add(Dropout(parameters['dropout']))
    model.add(Flatten())
    model.add(Dense(parameters['neuron'], activation=parameters['activation'],kernel_regularizer=regularizers.l2(0.02), name='FC4'))
    model.add(Dropout(parameters['dropout']))

    model.add(Dense(8, activation=parameters['activation'], name='predictions'))



    print('Done!\n')

    myoptim = optimizers.Adam(lr=parameters['lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    sgd = optimizers.SGD(lr=parameters['lr'], decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=myoptim,
                  metrics=['accuracy'])

    # this is the dataset configuration we will use for training
    # only rescaling
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2)


    # this is the dataset configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR + '/train',  # this is the target directory
        target_size=(IMG_SIZE, IMG_SIZE),  # all images will be resized to IMG_SIZExIMG_SIZE
        batch_size=parameters['batch_size'],
        classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        DATASET_DIR + '/test',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=parameters['batch_size'],
        classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        class_mode='categorical')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=1881 // parameters['batch_size'],
        epochs=NUM_EPOCHS / 2,
        validation_data=validation_generator,
        validation_steps=807 // parameters['batch_size'])


    return history, model


if __name__ == "__main__":

    if not os.path.exists(DATASET_DIR):
        print(Color.RED, 'ERROR: dataset directory ' + DATASET_DIR + ' do not exists!\n')
        quit()

    if os.path.exists(MODEL_FNAME):
        print('WARNING: model file ' + MODEL_FNAME + ' exists and will be overwritten!\n')

    # defining checkpoints and early stopping
    #file_path = "WEIGHTS_MODEL_scratch.hdf5"
    #callbacks = get_callbacks(filepath=file_path, patience=5)

    print('Start training...\n')
    from utils import *
    # this is the dataset configuration we will use for training
    # only rescaling
    train_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is the dataset configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR + '/train',  # this is the target directory
        target_size=(IMG_SIZE, IMG_SIZE),  # all images will be resized to IMG_SIZExIMG_SIZE
        batch_size=BATCH_SIZE,
        classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        DATASET_DIR + '/test',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        class_mode='categorical')

    # callbacks=callbacks)
    #'patience': [0.2, 0.4, 0.5, 0.6],

    X_train, Y_train = loadfromGenerator(train_generator)

    X_test, Y_test = loadfromGenerator(validation_generator)

    parameters = {'lr': (0.001, 0.01, 0.02, 0.1),
                  'neuron': [128, 248, 512, 64],
                  'filter_1':[10,20 ,30,40,50, 60, 70,80,90,100],
                  'filter_2': [10,20 ,30,40,50, 60, 70,80,90,100],
                  'filter_3': [10,20 ,30,40,50, 60, 70,80,90,100],
                  'kernel_size1':[3,5,7,10],
                  'kernel_size2':[3,5,7,10],
                  'kernel_size3': [3, 5, 7, 10],
                  'batch_size': [20, 30, 40],
                  'dropout': (0, 0.1, 0.3, 0.6),
                  'activation': ['relu', 'elu', 'selu','sigmoid']
                  }

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices

    X = np.asarray(X_train)
    Y = np.asarray(X_test)
    Y_test = np.asarray(Y_test)
    Y_train = np.asarray(Y_train)

    h = ta.Scan(x=X, y=Y_train,
                x_val=Y, y_val=Y_test,
                params=parameters,
                dataset_name='first_test',
                experiment_no='1',
                model=create_Model, grid_downsample=.1)


