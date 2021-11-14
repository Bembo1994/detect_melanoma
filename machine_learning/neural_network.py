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
        self.img_size = utils.IMG_SIZE_VGG
        #self.neural_network = VGG16(weights='imagenet', include_top=False, input_shape=(self.img_size, self.img_size, 3))
        #print(self.neural_network.summary())
        self.checkpoint_path = os.path.dirname(os.path.realpath(__file__)) + "/checkpoints/vgg16/"
        self.make_dirs(self.checkpoint_path)
        # self.model = self.get_unet_model(h, w, c)

    def make_dirs(self, path):
        if not os.path.exists(path):
            print("Make directory : {}".format(path))
            os.makedirs(path)

    def get_model(self):

        name_model = self.checkpoint_path + 'vgg16_model_tesla_CV_{}i_{}e_{}bs_{}lr_{}.hdf5'.format(
            utils.LIMIT_IMAGES_CLASSIFICATION_PKL, utils.EPOCHS_VGG, utils.BS_VGG, str(utils.LR_VGG).split(".")[1],
            utils.FUNCTION_VGG)

        name_logger = self.checkpoint_path + 'vgg16_training_logger_tesla_CV_{}i_{}e_{}bs_{}lr_{}.csv'.format(
            utils.LIMIT_IMAGES_CLASSIFICATION_PKL, utils.EPOCHS_VGG, utils.BS_VGG, str(utils.LR_VGG).split(".")[1],
            utils.FUNCTION_VGG)

        if os.path.isfile(name_model):  # and os.path.isfile(name_logger) :
            print("VGG16 is already train")
            return load_model(name_model)

        old_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(self.img_size, self.img_size, 3))

        # Freeze 0-14 the layers
        for layer in old_vgg.layers[:]:
            layer.trainable = False

        # Check the trainable status of the individual layers
        for layer in old_vgg.layers:
            print(layer, layer.trainable)

        print(old_vgg.summary())

        # Create the model
        model = keras.Sequential()

        # Add the vgg convolutional base model
        model.add(old_vgg)

        # Add new layers
        model.add(Flatten())
        model.add(Dense(4096, activation='softmax'))
        model.add(Dense(4096, activation='softmax'))
        model.add(Dense(1, activation='sigmoid'))  # sigmoid or relu

        # Show a summary of the model. Check the number of trainable parameters
        print(model.summary())

        opt = Adam(learning_rate=utils.LR_VGG)  # 0.001 default
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        print(model.summary())

        return model

    def fit_model(self, model, X_train, X_val, y_train, y_val, csv_logger):
        model.fit(X_train, y_train,
                  batch_size=utils.BS_VGG,
                  verbose=1,
                  epochs=utils.EPOCHS_VGG,
                  validation_data=(X_val, y_val),
                  shuffle=False,
                  callbacks=[csv_logger])

    def train(self, model, X_train, X_val, y_train, y_val):
        name_model=self.checkpoint_path + 'vgg16_model_tesla_CV_{}i_{}e_{}bs_{}lr_{}.hdf5'.format(
            utils.LIMIT_IMAGES_CLASSIFICATION_PKL, utils.EPOCHS_VGG, utils.BS_VGG, str(utils.LR_VGG).split(".")[1],
            utils.FUNCTION_VGG)

        name_logger = self.checkpoint_path + 'vgg16_training_logger_tesla_CV_{}i_{}e_{}bs_{}lr_{}.csv'.format(
            utils.LIMIT_IMAGES_CLASSIFICATION_PKL, utils.EPOCHS_VGG, utils.BS_VGG, str(utils.LR_VGG).split(".")[1],
            utils.FUNCTION_VGG)

        if os.path.isfile(name_model) and os.path.isfile(name_logger):
            print("UNet is already train with this params")
            return

        csv_logger = CSVLogger(name_logger)
        if tf.test.is_gpu_available():
            with tf.device('/device:GPU:0'):
                self.fit_model(model, X_train, X_val, y_train, y_val, csv_logger)
        else:
            self.fit_model(model, X_train, X_val, y_train, y_val, csv_logger)
        model.save(name_model)
        print("Model and csv logger are saved")

class UNet():

    def __init__(self):
        self.img_size = utils.IMG_SIZE_UNET
        self.img_channels = utils.IMG_CHANNELS_UNET
        self.checkpoint_path = os.path.dirname(os.path.realpath(__file__))+"/checkpoints/unet/"
        self.make_dirs(self.checkpoint_path)
        #self.model = self.get_unet_model(h, w, c)

    def make_dirs(self,path):
        if not os.path.exists(path):
            print("Make directory : {}".format(path))
            os.makedirs(path)

    def get_model(self, height, width, channels):
        name_model = self.checkpoint_path + 'unet_model_colab_{}i_{}e_{}bs_{}lr_{}.hdf5'.format(utils.LIMIT_IMAGES_SEGMENTATION_PKL, utils.EPOCHS_UNET, utils.BS_UNET, str(utils.LR_UNET).split(".")[1], utils.FUNCTION_UNET)
        name_logger = self.checkpoint_path + 'unet_training_logger_colab_{}i_{}e_{}bs_{}lr_{}.csv'.format(utils.LIMIT_IMAGES_SEGMENTATION_PKL, utils.EPOCHS_UNET, utils.BS_UNET, str(utils.LR_UNET).split(".")[1], utils.FUNCTION_UNET)

        if os.path.isfile(name_model) :#and os.path.isfile(name_logger) :
            print("UNet is already train")
            return load_model(name_model)

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

        outputs = Conv2D(1, (1, 1), activation=utils.FUNCTION_UNET)(c9) #prova con relu -> sigmoid

        model = Model(inputs=[inputs], outputs=[outputs])
        opt = Adam(learning_rate=utils.LR_UNET) # 0.001 default
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model

    def fit_model(self, model, X_train, X_val, y_train, y_val, csv_logger):
        model.fit(X_train, y_train,
                  batch_size=utils.BS_UNET,
                  verbose=1,
                  epochs=utils.EPOCHS_UNET,
                  validation_data=(X_val, y_val),
                  shuffle=False,
                  callbacks=[csv_logger])

    def train(self, model, X_train, X_val, y_train, y_val):
        name_model = self.checkpoint_path + 'unet_model_colab_{}i_{}e_{}bs_{}lr_{}.hdf5'.format(utils.LIMIT_IMAGES_SEGMENTATION_PKL, utils.EPOCHS_UNET, utils.BS_UNET, utils.LR_UNET.split(".")[1],utils.FUNCTION_UNET)
        name_logger = self.checkpoint_path+'unet_training_logger_colab_{}i_{}e_{}bs_{}lr_{}.csv'.format(utils.LIMIT_IMAGES_SEGMENTATION_PKL, utils.EPOCHS_UNET, utils.BS_UNET, utils.LR_UNET.split(".")[1], utils.FUNCTION_UNET)
        if os.path.isfile(name_model) and os.path.isfile(name_logger) :
            print("UNet is already train with this params")
            return

        csv_logger = CSVLogger(name_logger)
        if tensorflow.test.is_gpu_available():
            with tf.device('/device:GPU:0'):
                self.fit_model(model, X_train, X_val, y_train, y_val, csv_logger)
        else:
            self.fit_model(model, X_train, X_val, y_train, y_val, csv_logger)
        model.save(name_model)
        print("Model and csv logger are saved")