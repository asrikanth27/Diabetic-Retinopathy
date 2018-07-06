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

x_train = np.load("../x_trainleft224.npy")
y_train = np.load("../y_trainleft224.npy")

print("X-size: "+str(x_train.shape))
print("Sample shape: "+str(x_train[0].shape))
print("Y-size: "+str(y_train.shape))
print("Label shape: "+str(y_train[0].shape))
print("\nImages loading complete!!\nLoading model....")

from keras.models import load_model
model = load_model("convnet224x224x3_left_trained_t85_v67.h5")
print("Model loading complete!! Beginning training...")

from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau

filepath="../custom_net_checkpoint_left.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='auto')
changerate = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='auto', epsilon=0.01, cooldown=0, min_lr=1e-6)
callbacks_list = [checkpoint, changerate]

model.fit(x=x_train, y=y_train, batch_size=48, epochs=50,verbose=1, validation_split=0.1, shuffle=True, callbacks=callbacks_list)
print("\n\nTraining Complete!!!")

model.save("../custom_net_complete_left.h5")
print("Model saved")
