import os
import getpass


from utils import *
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape, Input,concatenate
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
import talos as ta
from keras.optimizers import Adam, Nadam
from keras.activations import softmax
from keras.losses import categorical_crossentropy, logcosh
import keras
import numpy as np
from keras.models import model_from_json

#user defined variables

IMG_SIZE    = 30
BATCH_SIZE  = 32
#num_epochs has to be an even number, because first iteration is training for num_epochs / 2
NUM_EPOCHS = 100
base_model_last_block_layer_number = 30
DATASET_DIR = '/home/grupo03/MIT_split_new'
MODEL_FNAME = 'xception_finetuning.h5'
W_FNAME='WEIGHTS_xception_finetuning.h5'
sgd = optimizers.SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
myoptim=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# serialize model to JSON
xception_model = xception.Xception(weights='imagenet', include_top = False)
model_json = xception_model.to_json()
with open("modelsaved.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
xception_model.save_weights("saved.h5")
print("Saved model to disk")

# later...



def getXceptionAugmentedModel(x_train, y_train, x_val, y_val,parameters):
    print('Building XCeption model...\n')

    #include_top: whether we want to include the fully-connected layer at the top of the network.
    # load json and create model
    json_file = open('modelsaved.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("saved.h5")
    print("Loaded model from disk")

    x = loaded_model.layers[-4].output

    # x = xception_model.get_layer('block5_pool').output

    x = GlobalMaxPooling2D()(x)

    x = Dense(parameters['first_neuron'], activation=parameters['activation'], name='fc2')(x)
    x = Dropout(parameters['dropout'])(x)
    x = Dense(parameters['first_neuron'], activation=parameters['activation'], name='fc3')(x)
    x = Dropout(parameters['dropout'])(x)
    x = Dense(8, activation='softmax',name='predictions')(x)

    model = Model(input=loaded_model.input, output=x)

    print(model.summary())
    plot_model(model, to_file= 'model_Xception_'+str(datetime.datetime.now())+'.png', show_shapes=True, show_layer_names=True)

    for layer in xception_model.layers[:]:
        layer.trainable = False
    #freezing the weights:

    print('Done!\n')

    model.compile(loss='categorical_crossentropy',
                  optimizer=myoptim,
                  metrics=['accuracy'])

    history1 = model.fit(
        x_train, y_train,
        steps_per_epoch=parameters['epochs'] // parameters['batch_size'],
        epochs=parameters['epochs']//2,
        validation_data=[x_val, y_val],
        validation_steps=2288 // parameters['batch_size'],
      

    )

    for layer in model.layers[:base_model_last_block_layer_number]:
        layer.trainable = False
    for layer in model.layers[base_model_last_block_layer_number:]:
        layer.trainable = True
    print("recompiling the model with new layer training rules...")

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data

    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        x_train, y_train,
        steps_per_epoch=parameters['epochs'] // parameters['batch_size'],
        epochs=parameters['epochs']//2,
        validation_data=[x_val, y_val],
        validation_steps= 2288 // parameters['batch_size'],
 
    )

    return history,model


if __name__ == "__main__":

    if not os.path.exists(DATASET_DIR):
        print(Color.RED, 'ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
        quit()


    if os.path.exists(MODEL_FNAME):
        print('WARNING: model file '+MODEL_FNAME+' exists and will be overwritten!\n')

    #defining checkpoints and early stopping
    file_path = "aug_model_weights.hdf5"
    callbacks = get_callbacks(filepath=file_path, patience=5)

    from utils import *
    import numpy as np

    parameters = {'lr': (0.001, 0.01, 0.02 ,0.1),
         'first_neuron': [ 128,248,512, 1024],
         'batch_size': [20, 30, 40],
         'epochs': [  40, 100,200, 400],
         'dropout': (0, 0.1, 0.3, 0.6),
         'activation': ['relu', 'elu' ,'selu']
         }
    # only rescaling
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
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


    def getXYfromGenerator(generator):
        X, Y = generator.next()
        batch_index = 1

        while batch_index <= generator.batch_index:
            auxX, auxY = generator.next()
            X = np.concatenate((X, auxX))
            Y = np.concatenate((Y, auxY))
            batch_index = batch_index + 1

        return X, Y


    X_train, Y_train = getXYfromGenerator(train_generator)

    X_test, Y_test = getXYfromGenerator(validation_generator)

    img_rows, img_cols = 30, 30
    num_classes = 8


    input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
 

    X=np.asarray(X_train)
    Y=np.asarray(X_test)
    Y_test=np.asarray(Y_test)
    Y_train=np.asarray(Y_train)

    h = ta.Scan(x=X,y=Y_train,
                x_val=Y,y_val= Y_test,
                params=parameters,
                dataset_name='first_test',
                experiment_no='1',
                model=getXceptionAugmentedModel,grid_downsample=.01)

    print('Best Accuracy')
    print(h.high('acc'))
    print('Best parameters')
    print(h.best_params())
