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
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger
import os
import tensorflow as tf
import numpy as np
from utils import utils
import pandas as pd

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

class UNet():

    def __init__(self):
        self.img_size = utils.IMG_SIZE_UNET
        self.img_channels = utils.IMG_CHANNELS_UNET
        self.checkpoint_path = os.path.dirname(os.path.realpath(__file__))+"/checkpoints/"
        self.make_dirs(self.checkpoint_path)
        #self.model = self.get_unet_model(h, w, c)

    def make_dirs(self,path):
        if not os.path.exists(path):
            print("Make directory : {}".format(path))
            os.makedirs(path)

    def get_model(self, height, width, channels):

        if os.path.isfile(self.checkpoint_path + "unet_model.hdf5"):
            return load_model(self.checkpoint_path + "unet_model.hdf5")

        # Build the model
        inputs = Input((height, width, channels))
        # s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
        s = inputs

        # Contraction path
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        # Expansive path
        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = Conv2D(1, (1, 1), activation='relu')(c9) #prova con relu -> sigmoid

        model = Model(inputs=[inputs], outputs=[outputs])
        opt = Adam(learning_rate=utils.LR_UNET) # 0.001 default
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        print(model.summary())

        return model

    def train_and_get_history(self, model, X_train, X_test, y_train, y_test):
        if os.path.isfile(self.checkpoint_path+'unet_training_logger.csv'):
            return pd.read_csv(self.checkpoint_path+'unet_training_logger.csv')

        csv_logger = CSVLogger(self.checkpoint_path+'unet_training_logger.csv')

        history = model.fit(X_train, y_train,
                                 batch_size=utils.BS_UNET,
                                 verbose=1,
                                 epochs=utils.EPOCHS_UNET,
                                 validation_data=(X_test, y_test),
                                 shuffle=False,
                                 callbacks=[csv_logger])

        model.save(self.checkpoint_path + "unet_model.hdf5")

        return history