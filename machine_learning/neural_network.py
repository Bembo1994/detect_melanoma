from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
from keras.utils.vis_utils import plot_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
from tensorflow import keras
import os
import tensorflow as tf
import numpy as np
from utils import utils

class NN_VGG16():

    def __init__(self):
        self.img_size = utils.IMG_SIZE
        self.neural_network = VGG16(weights='imagenet', include_top=False, input_shape=(self.img_size, self.img_size, 3))
        print(self.neural_network.summary())

    def freeze_neural_network(self):
        # Freeze 0-14 the layers
        for layer in self.neural_network.layers[:]:
            layer.trainable = False

        # Check the trainable status of the individual layers
        for layer in self.neural_network.layers:
            print(layer, layer.trainable)

    def add_top_layers(self):
        # Create the model
        new_model = keras.Sequential()

        # Add the vgg convolutional base model
        new_model.add(self.neural_network)

        # Add new layers
        new_model.add(Flatten())
        new_model.add(Dense(4096, activation='softmax'))
        new_model.add(Dense(4096, activation='softmax'))
        new_model.add(Dense(1, activation='sigmoid'))

        # Show a summary of the model. Check the number of trainable parameters
        print(new_model.summary())

        return new_model

    def compile(self, nn):
        nn.compile(optimizer='adam',
                          loss="binary_crossentropy",
                          metrics=['accuracy'])

    def train(self,nn,ds_train,ds_val):
        history = nn.fit(
            ds_train,
            validation_data=ds_val,
            batch_size=16,
            epochs=utils.EPOCHS,
            verbose=1
        )
        return history