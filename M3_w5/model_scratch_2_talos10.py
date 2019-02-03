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

from talos.model.early_stopper import early_stopper
from talos.model.normalizers import lr_normalizer
from keras.losses import categorical_crossentropy, logcosh
from keras.optimizers import Adam, Nadam, SGD,Adamax
from keras.activations import relu, elu,selu
from keras.models import model_from_json
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Flatten, Activation



def create_Model(x_train, y_train, x_val, y_val, parameters):




    model = Sequential()
    model.add(BatchNormalization(input_shape=x_train.shape[1:]))
    model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


    model.add(BatchNormalization(input_shape=x_train.shape[1:]))
    model.add(Conv2D(60, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(8))
    model.add(Activation('softmax'))

    print('Done!\n')

    """myoptim = optimizers.Adam(lr=parameters['lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    sgd = optimizers.SGD(lr=parameters['lr'], decay=1e-6, momentum=0.9, nesterov=True)"""

    model.compile(optimizer=SGD(lr=lr_normalizer(0.1, SGD)),
                  loss=parameters['loss'],
                  metrics=['acc'])



    file_path = "aug_model_weights.hdf5"
    #callbacks = get_callbacks(filepath=file_path, patience=parameters['patience'])
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
        batch_size=20,
        classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        DATASET_DIR + '/test',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=20,
        classes=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        class_mode='categorical')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=1881 // 20,
        epochs=80// 2,
        validation_data=validation_generator,
        validation_steps=807 // 20)
  

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
    # this is the dataset configuration we will use for training
    # only rescaling
    train_datagen = ImageDataGenerator(
        rescale=1. / 255
        )


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

    parameters = { 'lr':[0.001, 0.01,0.02],
                  'first_neuron': [64,128, 248, 512],
                  'batch_size': [32],
                  'dropout': [0, 0.1, 0.3],
                  'activation': ['relu','selu','elu'],
                  'optimizer': [SGD,Adam,Adamax],
                  'epochs':[100],
                  'loss':[categorical_crossentropy]
                  }

    p = {'activation': [relu, elu],
         'optimizer': ['Nadam', 'Adam'],
         'hidden_layers': [0, 1, 2],
         'batch_size': [20, 30, 40],
         'epochs': [200]}

    parameters_good={ 'lr':[0.01],
                  'first_neuron': [512],
                  'batch_size': [20],
                  'dropout': [0],

                  'optimizer': [SGD],
                  'epochs':[80],
                  'loss':[categorical_crossentropy]
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

    history,model=create_Model(X, Y_train, Y, Y_test, parameters_good)

    """h = ta.Scan(x=X, y=Y_train,
                x_val=Y, y_val=Y_test,
                params=parameters,
                dataset_name='test',
                experiment_no='6.4',
                model=create_Model, grid_downsample=.01)

    ta.Deploy(h, 'model_scratch2')"""
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
