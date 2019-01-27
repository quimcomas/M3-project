import os
import getpass

from utils import *
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape, Input, concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import datetime
from keras.layers import Dropout
from keras import regularizers
from keras import optimizers
from keras.applications import xception
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as k

# user defined variables
IMG_SIZE = 30
BATCH_SIZE = 32
# num_epochs has to be an even number, because first iteration is training for num_epochs / 2
NUM_EPOCHS = 100
base_model_last_block_layer_number = 30
DATASET_DIR = '/home/grupo03/MIT_split_new'
MODEL_FNAME = 'xception_finetuning.h5'
W_FNAME = 'WEIGHTS_xception_finetuning.h5'
sgd = optimizers.SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
myoptim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


def getXceptionAugmentedModel():
    print('Building XCeption model...\n')

    # include_top: whether we want to include the fully-connected layer at the top of the network.
    xception_model = xception.Xception(weights='imagenet', include_top=False)

    x = xception_model.layers[-4].output

    # x = xception_model.get_layer('block5_pool').output

    x = GlobalMaxPooling2D()(x)

    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu', name='fc3')(x)
    x = Dropout(0.3)(x)
    x = Dense(8, activation='softmax', name='predictions')(x)

    model = Model(input=xception_model.input, output=x)

    print(model.summary())
    plot_model(model, to_file='model_Xception_' + str(datetime.datetime.now()) + '.png', show_shapes=True,
               show_layer_names=True)

    # freezing the weights:
    for layer in xception_model.layers[:]:
        layer.trainable = False

    print('Done!\n')

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
    file_path = "aug_model_weights.hdf5"
    callbacks = get_callbacks(filepath=file_path, patience=5)

    model = getXceptionAugmentedModel()
    print('Start training...\n')

    # this is the dataset configuration we will use for training
    # only rescaling
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.3
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

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=400 // BATCH_SIZE,
        epochs=NUM_EPOCHS / 2,
        validation_data=validation_generator,
        validation_steps=2288 // BATCH_SIZE,
        callbacks=callbacks)

    # Getting the Best Model
    model.load_weights(filepath=file_path)

    print('Done!\n')
    print('Saving the model into ' + MODEL_FNAME + ' \n')
    model.save_weights(W_FNAME)  # always save your weights after training or during training
    print('Done!\n')

    print("Start fine-tuning...")

    # based_model_last_block_layer_number: points to the layer in your model you want to train.
    # For example if you want to train the last block of a 19 layer VGG16 model this should be 15
    # If you want to train the last Two blocks of an Inception model it should be 172
    # layers before this number will use the pre-trained weights, layers above and including this number
    # will be re-trained based on the new data.
    for layer in model.layers[:base_model_last_block_layer_number]:
        layer.trainable = False
    for layer in model.layers[base_model_last_block_layer_number:]:
        layer.trainable = True
    print("recompiling the model with new layer training rules...")
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=400 // BATCH_SIZE,
        epochs=NUM_EPOCHS / 2,
        validation_data=validation_generator,
        validation_steps=2288 // BATCH_SIZE,
        callbacks=callbacks)

    # Getting the Best Model
    model.load_weights(filepath=file_path)

    print('Done!\n')
    print('Saving the model into ' + MODEL_FNAME + ' \n')
    model.save_weights(W_FNAME)  # always save your weights after training or during training
    print('Done!\n')

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('accuracy_xception_ft' + MODEL_FNAME + '_' + str(datetime.datetime.now()) + '.jpg')
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('loss_xception_ft' + MODEL_FNAME + '_' + str(datetime.datetime.now()) + '.jpg')

    print('Done!')
    # clearing memory:
    k.clear_session()
