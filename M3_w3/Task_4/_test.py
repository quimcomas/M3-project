import os
import getpass


from utils import *
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from PIL import Image
import numpy as np
from build_svm import *
from scipy.misc import imresize
from sklearn import svm


#PARAMETERS
IMG_SIZE    = 32
BATCH_SIZE  = 16
DATASET_DIR = '/home/mcv/datasets/MIT_split'
MODEL_FNAME = 'my_first_mlp.h5'
KERNEL='linear'





if not os.path.exists(DATASET_DIR):
  print(Color.RED, 'ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
  quit()


print('Building MLP model...\n')

#Build the Multi Layer Perceptron model
model = Sequential()
model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),input_shape=(IMG_SIZE, IMG_SIZE, 3),name='first'))
model.add(Dense(units=2048, activation='relu',name='second'))
#model.add(Dense(units=1024, activation='relu'))
model.add(Dense(units=8, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

print(model.summary())
plot_model(model, to_file='modelMLP.png', show_shapes=True, show_layer_names=True)

print('Done!\n')

if os.path.exists(MODEL_FNAME):
  print('WARNING: model file '+MODEL_FNAME+' exists and will be overwritten!\n')

print('Start training...\n')

# this is the dataset configuration we will use for training
# only rescaling
train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True)

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
#model.save_weights(MODEL_FNAME)  # always save your weights after training or during training
print('Done!\n')

  # summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
print('holiii vull plots\n')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('accuracy' + MODEL_FNAME + '.jpg')
plt.close()
  # summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss' + MODEL_FNAME + '.jpg')

# SVM classifcation using the output of last layer

model_layer = Model(inputs=model.input, outputs=model.get_layer('second').output)

directory_train=DATASET_DIR+ '/train/'
train_data,labels_train =load_data(IMG_SIZE,directory_train)
train_data=np.array(train_data)
labels_train=np.array(labels_train)

directory_test=DATASET_DIR+ '/test/'
test_data,labels_test =load_data(IMG_SIZE,directory_test)
test_data=np.array(test_data)
labels_test2=np.array(labels_test)

train_features=[]
i=0
for image in train_data:
    
    
    train_features.append(model_layer.predict(np.array(image))[0])
    i=i+1

test_features=[]
i=0
for imaget in test_data:
    test_features.append(model_layer.predict(np.array(imaget))[0])
    i=i+1
    
train_features=np.array(train_features)
test_features=np.array(test_features)
print(np.shape(train_features))

print('SVM-CrossValidation')
build_svm_kernel_crossvalidation(train_features[:].tolist(),test_features[:].tolist(),labels_train[:].tolist(),labels_test2[:].tolist())

"""KERNEL='rbf'

if KERNEL=='linear':
    clf= svm.SVC(kernel='linear',C=0.01,gamma=0.002).fit(train_features[:].tolist(), labels_train[:].tolist())

if KERNEL=='rbf':
    clf = svm.SVC(kernel='rbf', C=1, gamma=0.002).fit(train_features[:].tolist(), labels_train[:].tolist())

if KERNEL=='sigmoid':
    clf= svm.SVC(kernel='sigmoid',C=1,gamma=0.002).fit(train_features[:].tolist(), labels_train[:].tolist())


predicted= clf.predict(test_features[:].tolist())

accuracy = accuracy_score(labels_test2[:].tolist(), predicted, normalize=True)

print(accuracy)"""
