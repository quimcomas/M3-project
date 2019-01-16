from keras import Sequential
from keras.layers import Reshape, Dense
from keras.utils import plot_model
from utils import colorprint, Color


def create_MLP(IMG_SIZE):

    colorprint(Color.RED, 'Building MLP model...\n')

    #Build the Multi Layer Perceptron model
    model = Sequential()
    model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),input_shape=(IMG_SIZE, IMG_SIZE, 3),name='first'))
    model.add(Dense(units=2048, activation='relu',name='second'))
    #model.add(Dense(units=1024, activation='relu'))
    model.add(Dense(units=8, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    model.summary()

    plot_model(model, to_file='modelMLP.png', show_shapes=True, show_layer_names=True)

    colorprint(Color.RED,'Done! \n')

    return model