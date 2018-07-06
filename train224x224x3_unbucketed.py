import numpy as np
from PIL import Image
import os
import keras
from keras.utils import np_utils
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=1.0
set_session(tf.Session(config=config))

print("Imports done!\n")
print("Loading images...")

x_train = np.load("../x_train.npy")
y_train = np.load("../y_train.npy")

print("X-size: "+str(x_train.shape))
print("Sample shape: "+str(x_train[0].shape))
print("Y-size: "+str(y_train.shape))
print("Label shape: "+str(y_train[0].shape))
print("\nImages loading complete!!\nLoading model....")

from keras.models import load_model
model = load_model("convnet224x224x3_untrained.h5")
print("Model loading complete!! Beginning training...")

from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau

filepath="../convnet_checkpoint_unbucketed.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
changerate = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='auto', epsilon=0.01, cooldown=0, min_lr=1e-6)
callbacks_list = [checkpoint, changerate]

history = model.fit(x=x_train, y=y_train, batch_size=96, epochs=20,verbose=1, validation_split=0.1, shuffle=True, callbacks=callbacks_list)
print("\n\nTraining Complete!!!")

model.save("../convnet_complete_unbucketed.h5")
print("Model saved")

import matplotlib.pyplot as plt
plt.switch_backend('agg')
#  "Accuracy"
train_acc = history.history['acc']
val_acc = history.history['val_acc']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.subplot(211)
plt.plot(train_acc)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#Loss
plt.subplot(212)
plt.plot(train_loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss'], loc='upper left')

fig_name = "fig_unbucketed"

plt.savefig("../"+fig_name+".png")

np.save(fig_name+"_trainacc.npy", np.asarray(train_acc))
np.save(fig_name+"_valacc.npy", np.asarray(val_acc))
np.save(fig_name+"_trainloss.npy", np.asarray(train_loss))
np.save(fig_name+"_valloss.npy", np.asarray(val_loss))
print("Data saved!!")
