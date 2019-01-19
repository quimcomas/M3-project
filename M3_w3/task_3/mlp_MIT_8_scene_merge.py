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


#user defined variables
IMG_SIZE    = 30
BATCH_SIZE  = 16
DATASET_DIR = '/home/mcv/datasets/MIT_split'
MODEL_FNAME = 'my_first_mlp.h5'
W_FNAME='WEIGHTS_my_first_mlp.h5'


if not os.path.exists(DATASET_DIR):
  print(Color.RED, 'ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
  quit()


print('Building MLP model...\n')

#Build the MLP using merging multiple outputs (3 layers)

input_1 = Input(shape=(IMG_SIZE, IMG_SIZE, 3),dtype='float32',name= 'Input')

reshape = Reshape((IMG_SIZE*IMG_SIZE*3,), input_shape=(IMG_SIZE, IMG_SIZE, 3),name='Reshape')(input_1)

left = Dense(units=1024,  activation='relu', kernel_regularizer=regularizers.l2(0.05), name='Left_layer')(reshape)

center = Dense(units=1024,  activation='relu', kernel_regularizer=regularizers.l2(0.05), name='Center_layer')(reshape)

right = Dense(units=1024,  activation='relu', kernel_regularizer=regularizers.l2(0.05), name='Right_layer' )(reshape)

merged = concatenate([left, center, right], name='Merged_layer')

final = Dense(units=8, activation='softmax', name='Final_layer')(merged)

model = Model(inputs=input_1, outputs=final)

sgd = optimizers.SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

print(model.summary())
plot_model(model, to_file= 'outputs/modelMLP_merge '+str(datetime.datetime.now())+'.png', show_shapes=True, show_layer_names=True)

print('Done!\n')

if os.path.exists(MODEL_FNAME):
  print('WARNING: model file '+MODEL_FNAME+' exists and will be overwritten!\n')

print('Start training...\n')

# this is the dataset configuration we will use for training
# only rescaling
train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True

)

# this is the dataset configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        DATASET_DIR+'/train',  # this is the target directory
        target_size=(IMG_SIZE, IMG_SIZE),  # all images will be resized to IMG_SIZExIMG_SIZE
        batch_size=BATCH_SIZE,
        classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
        class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        DATASET_DIR+'/test',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
        class_mode='categorical')

history = model.fit_generator(
        train_generator,
        steps_per_epoch=1881 // BATCH_SIZE,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=807 // BATCH_SIZE)

print('Done!\n')
print('Saving the model into '+MODEL_FNAME+' \n')
model.save_weights(W_FNAME)  # always save your weights after training or during training
print('Done!\n')

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('outputs/accuracy_merge' + MODEL_FNAME + '_' + str(datetime.datetime.now()) + '.jpg')
plt.close()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('outputs/loss_merge' + MODEL_FNAME + '_' + str(datetime.datetime.now()) + '.jpg')

print('Done!')
