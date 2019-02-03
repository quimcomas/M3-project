import os
import getpass
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.parse import urlencode
import numpy as np
from utils import *
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape, Input, concatenate, \
    MaxPooling2D, Conv2D, InputLayer, DepthwiseConv2D, ZeroPadding2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import ReLU, PReLU
from keras.utils import plot_model
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import datetime
from keras.layers import Dropout
from keras import regularizers
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras import backend as k
import talos as ta

# user defined variables
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 50
DATASET_DIR = '/home/mcv/datasets/MIT_split'
MODEL_FNAME = 'MODEL_scratch.h5'
W_FNAME = 'WEIGHTS_MODEL_scratch.h5'


def create_Model():
    print('Building our scratch model...\n')

    model = Sequential()

    model.add(InputLayer(input_shape=[IMG_SIZE,IMG_SIZE, 3], name='Input'))
    model.add(ZeroPadding2D(padding=(1, 1),input_shape=(3,224,224)))
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding='same', activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(DepthwiseConv2D((3, 3), padding='valid', depth_multiplier=1, activation='relu', strides=(1, 1)))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='same',activation='relu', use_bias=False))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(128, (5, 5), strides=(1, 1), padding='same',activation='relu', use_bias=False))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(512, activation=ReLU(), kernel_regularizer=regularizers.l2(0.03), name='FC4'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='softmax', name='predictions'))

    print(model.summary())
    plot_model(model, to_file='model_scratch_' + str(datetime.datetime.now()) + '.png', show_shapes=True,
               show_layer_names=True)

    print('Done!\n')

    myoptim = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy',
                  optimizer=myoptim,
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":

    if not os.path.exists(DATASET_DIR):
        print(Color.RED, 'ERROR: dataset directory ' + DATASET_DIR + ' do not exists!\n')
        quit()

    if os.path.exists(MODEL_FNAME):
        print('WARNING: model file ' + MODEL_FNAME + ' exists and will be overwritten!\n')

    # defining checkpoints and early stopping
    # file_path = "WEIGHTS_MODEL_scratch.hdf5"
    # callbacks = get_callbacks(filepath=file_path, patience=5)

    model = create_Model()

    print('Start training...\n')

    # this is the dataset configuration we will use for training
    # only rescaling
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        shear_range=0.,
        zoom_range=0.1,
        fill_mode='nearest',
        width_shift_range=0.1,
        height_shift_range=0.1)

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

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=1881 // BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=807 // BATCH_SIZE)
    # callbacks=callbacks)

    # model.load_weights(filepath=file_path)

    print('Done!\n')
    print('Saving the model into ' + MODEL_FNAME + ' \n')
    model.save_weights(MODEL_FNAME)  # always save your weights after training or during training
    print('Done!\n')

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('accuracy_scratch_ft' + MODEL_FNAME + '_' + str(datetime.datetime.now()) + '.jpg')
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('loss_scratch_ft' + MODEL_FNAME + '_' + str(datetime.datetime.now()) + '.jpg')

    print('Done!')
    # clearing memory:
    k.clear_session()
