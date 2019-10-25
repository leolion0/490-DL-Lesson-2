from copy import deepcopy

from keras import Sequential
from keras.datasets import mnist
import numpy as np
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt

EPOCHS = 5
VERB = 2
H_LAYERS = 2
ACT = 'relu'

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape[1:])
# process the data
# 1. convert each image of shape 28*28 to 784 dimensional which will be fed to the network as a single feature
dimData = np.prod(train_images.shape[1:])
print(dimData)
train_data = train_images.reshape(train_images.shape[0], dimData)
test_data = test_images.reshape(test_images.shape[0], dimData)

# convert data to float and scale values between 0 and 1
train_data = train_data.astype('float')
test_data = test_data.astype('float')
# scale data
train_data /= 255.0
test_data /= 255.0
train_data_unscaled = train_data * 255.0
test_data_unscaled = test_data * 255.0
# change the labels frominteger to one-hot encoding. to_categorical is doing the same thing as LabelEncoder()
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

# creating network
model = Sequential()

model2 = deepcopy(model)
model.add(Dense(512, activation='relu', input_shape=(dimData,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model2.add(Dense(512, activation='tanh', input_shape=(dimData,)))
for i in range(H_LAYERS - 1):
    model2.add(Dense(512, activation=ACT))
model2.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=EPOCHS, verbose=VERB,
                    validation_data=(test_data, test_labels_one_hot))

model2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history2 = model2.fit(train_data_unscaled, train_labels_one_hot, batch_size=256, epochs=EPOCHS, verbose=VERB,
                      validation_data=(test_data_unscaled, test_labels_one_hot))

print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')


plt.plot(history.history['acc'])
plt.plot(history2.history['acc'])
plt.title('Model accuracy TRAIN')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['TrainA', 'TrainB'], loc='upper left')
plt.show()

plt.plot(history.history['val_acc'])
plt.plot(history2.history['val_acc'])
plt.title('Model accuracy TEST')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['TestA', 'TestB'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history2.history['loss'])
plt.title('Model loss TRAIN')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['TrainA', 'TrainB'], loc='upper left')
plt.show()

plt.plot(history.history['val_loss'])
plt.plot(history2.history['val_loss'])
plt.title('Model loss TEST')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['TestA', 'TestB'], loc='upper left')
plt.show()
