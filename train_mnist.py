from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import datetime

# setting input image dimensions as tensorflow
K.set_image_dim_ordering('tf')

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# print(X_train[0].shape)
# print(y_train.shape)
# print(y_train)

#plotting mnist
# plt.subplot(221)
# plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
# plt.subplot(222)
# plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
# plt.subplot(223)
# plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
# plt.subplot(224)
# plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# plt.show()

# setting the seed
seed = 7
numpy.random.seed(seed)

# coverting the image

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# normalizing the grayscale image (0-255) --> (0-1)
X_train /= 255
X_test /= 255

# OneHotEncoding the classes of train and test data
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)
num_classes = y_test.shape[1]


# Simple CNN model
def CNN_classifier():
    model = Sequential()
    model.add(Conv2D(32, (5,5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    #compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    model = CNN_classifier()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print('Accuracy: {}%'.format(scores[1]*100))
    print('Loss: {}'.format(scores[0]))

    print('Saving Model')
    tot = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    filen = 'mnist_cnn_'+tot+'.h5'
    model.save(filen)
    print('Completed')
