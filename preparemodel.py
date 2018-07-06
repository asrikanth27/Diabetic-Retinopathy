import keras
from keras.layers import Conv2D, MaxPooling2D, Input, AveragePooling2D
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

print("Creating model...")
model = Sequential()

model.add(Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(224,224,3)))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
#111x111x64

model.add(Conv2D(128, (1,1), padding='same', activation='relu'))
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
#55x55x128

model.add(Conv2D(256, (1,1), padding='same', activation='relu'))
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
#27x27x256

model.add(Conv2D(512, (1,1), padding='same', activation='relu'))
model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
model.add(Conv2D(512, (5,5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
#13x13x512

model.add(Conv2D(1024, (1,1), padding='same', activation='relu'))
model.add(Conv2D(1024, (3,3), padding='same', activation='relu'))
model.add(Conv2D(1024, (5,5), padding='same', activation='relu'))
#model.add(Conv2D(2048, (7,7), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
#6x6x1024

model.add(Conv2D(1256, (3,3), padding='same', activation='relu'))
model.add(Conv2D(1256, (1,1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
#2x2x1256

model.add(AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
#1x1x1256

model.add(Flatten())

#model.add(Dense(4096, activation='relu', use_bias=True))
model.add(Dense(628, activation='relu', use_bias=True))
model.add(Dense(314, activation='relu', use_bias=True))
model.add(Dense(5, activation='softmax', use_bias=True))

print("Model created!!")
model.summary()
print("\nCompiling model...")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



sgd = SGD(momentum=0.9, nesterov=True, lr=0.003)
#callbacks = [LearningRateScheduler(8, verbose=1)]
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
print("Successfully compiled model!!")

model.save("WorkingModels/convnet224x224x3_untrained.h5")
print("Model saved!!")
