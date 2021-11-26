from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
from tensorflow import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalMaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger
import os
import tensorflow as tf
import numpy as np
from utils import utils
import pandas as pd

class NN_ResNet50():

    def __init__(self):
        self.img_size = utils.IMG_SIZE_RESNET
        self.checkpoint_path = os.path.dirname(os.path.realpath(__file__)) + "/checkpoints/resnet50/"
        self.make_dirs(self.checkpoint_path)

    def make_dirs(self, path):
        if not os.path.exists(path):
            print("Make directory : {}".format(path))
            os.makedirs(path)

    def get_model(self, cv_or_unet_preprocessing):

        name_model = self.checkpoint_path + 'resnet_model_{}_{}e_{}bs_{}lr_{}.hdf5'.format(cv_or_unet_preprocessing,
            utils.EPOCHS_RESNET, utils.BS_RESNET, str(utils.LR_RESNET).split(".")[1],
            utils.FUNCTION_RESNET)

        name_logger = self.checkpoint_path + 'resnet_training_logger_{}_{}e_{}bs_{}lr_{}.csv'.format(cv_or_unet_preprocessing,
            utils.EPOCHS_RESNET, utils.BS_RESNET, str(utils.LR_RESNET).split(".")[1],
            utils.FUNCTION_RESNET)

        if os.path.isfile(name_model) and os.path.isfile(name_logger) :
            print("ResNet50 is already train")
            return load_model(name_model)

        base_model = ResNet50(weights='imagenet', include_top=False)

        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalMaxPooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(1, activation=utils.FUNCTION_RESNET)(x)
        model = Model(inputs=base_model.input, outputs=x)
        print(model.summary())
        adam = Adam(lr=utils.LR_RESNET)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def fit_model(self, model, X_train, X_val, y_train, y_val, csv_logger, model_checkpoint_callback):
        model.fit(X_train,y_train,
                  batch_size=utils.BS_RESNET,
                  verbose=1,
                  epochs=utils.EPOCHS_RESNET,
                  validation_data=(X_val, y_val),
                  shuffle=True,
                  callbacks=[csv_logger, model_checkpoint_callback])

    def train(self, model, X_train, X_val, y_train, y_val, cv_or_unet_preprocessing):
        name_model = self.checkpoint_path + 'resnet_model_{}_{}e_{}bs_{}lr_{}.hdf5'.format(cv_or_unet_preprocessing,
            utils.EPOCHS_RESNET, utils.BS_RESNET, str(utils.LR_RESNET).split(".")[1],
            utils.FUNCTION_RESNET)

        name_logger = self.checkpoint_path + 'resnet_training_logger_{}_{}e_{}bs_{}lr_{}.csv'.format(cv_or_unet_preprocessing,
            utils.EPOCHS_RESNET, utils.BS_RESNET, str(utils.LR_RESNET).split(".")[1],
            utils.FUNCTION_RESNET)

        if os.path.isfile(name_model) and os.path.isfile(name_logger):
            print("ResNet50 is already train with this params")
            return

        csv_logger = CSVLogger(name_logger)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        if tf.test.is_gpu_available():
            with tf.device('/device:GPU:0'):
                print("Work with GPU")
                self.fit_model(model, X_train, X_val, y_train, y_val, csv_logger, model_checkpoint_callback)
        else:
            print("Work with CPU")
            self.fit_model(model, X_train, X_val, y_train, y_val, csv_logger, model_checkpoint_callback)
        model.save(name_model)
        print("Model and csv logger are saved")

class NN_VGG16():

    def __init__(self):
        self.img_size = utils.IMG_SIZE_VGG
        self.checkpoint_path = os.path.dirname(os.path.realpath(__file__)) + "/checkpoints/vgg16/"
        self.make_dirs(self.checkpoint_path)


    def make_dirs(self, path):
        if not os.path.exists(path):
            print("Make directory : {}".format(path))
            os.makedirs(path)

    def get_model(self, cv_or_unet_preprocessing):

        name_model = self.checkpoint_path + 'vgg16_model_{}_{}e_{}bs_{}lr_{}.hdf5'.format(cv_or_unet_preprocessing,
            utils.EPOCHS_VGG, utils.BS_VGG, str(utils.LR_VGG).split(".")[1],
            utils.FUNCTION_VGG)

        name_logger = self.checkpoint_path + 'vgg16_training_logger_{}_{}e_{}bs_{}lr_{}.csv'.format(cv_or_unet_preprocessing,
            utils.EPOCHS_VGG, utils.BS_VGG, str(utils.LR_VGG).split(".")[1],
            utils.FUNCTION_VGG)

        if os.path.isfile(name_model) and os.path.isfile(name_logger) :
            print("NET is already train")
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

    def fit_model(self, model, X_train, X_val, y_train, y_val, csv_logger, model_checkpoint_callback):
        model.fit(X_train,y_train,
                  batch_size=utils.BS_VGG,
                  verbose=1,
                  epochs=utils.EPOCHS_VGG,
                  validation_data=(X_val, y_val),
                  shuffle=True,
                  callbacks=[csv_logger, model_checkpoint_callback])

    def train(self, model, X_train, X_val, y_train, y_val, cv_or_unet_preprocessing):
        name_model = self.checkpoint_path + 'vgg16_model_{}_{}e_{}bs_{}lr_{}.hdf5'.format(cv_or_unet_preprocessing,
            utils.EPOCHS_VGG, utils.BS_VGG, str(utils.LR_VGG).split(".")[1],
            utils.FUNCTION_VGG)

        name_logger = self.checkpoint_path + 'vgg16_training_logger_{}_{}e_{}bs_{}lr_{}.csv'.format(cv_or_unet_preprocessing,
            utils.EPOCHS_VGG, utils.BS_VGG, str(utils.LR_VGG).split(".")[1],
            utils.FUNCTION_VGG)

        if os.path.isfile(name_model) and os.path.isfile(name_logger):
            print("VGG is already train with this params")
            return

        csv_logger = CSVLogger(name_logger)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        if tf.test.is_gpu_available():
            with tf.device('/device:GPU:0'):
                self.fit_model(model, X_train, X_val, y_train, y_val, csv_logger, model_checkpoint_callback)
        else:
            self.fit_model(model, X_train, X_val, y_train, y_val, csv_logger, model_checkpoint_callback)
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
        name_model = self.checkpoint_path + 'unet_model_{}i_{}e_{}bs_{}lr_{}.hdf5'.format(utils.LIMIT_IMAGES_SEGMENTATION_PKL, utils.EPOCHS_UNET, utils.BS_UNET, str(utils.LR_UNET).split(".")[1], utils.FUNCTION_UNET)
        name_logger = self.checkpoint_path + 'unet_training_logger_{}i_{}e_{}bs_{}lr_{}.csv'.format(utils.LIMIT_IMAGES_SEGMENTATION_PKL, utils.EPOCHS_UNET, utils.BS_UNET, str(utils.LR_UNET).split(".")[1], utils.FUNCTION_UNET)

        if os.path.isfile(name_model) and os.path.isfile(name_logger) :
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
                  shuffle=True,
                  callbacks=[csv_logger])

    def train(self, model, X_train, X_val, y_train, y_val):
        name_model = self.checkpoint_path + 'unet_model_{}i_{}e_{}bs_{}lr_{}.hdf5'.format(utils.LIMIT_IMAGES_SEGMENTATION_PKL, utils.EPOCHS_UNET, utils.BS_UNET, str(utils.LR_UNET).split(".")[1], utils.FUNCTION_UNET)
        name_logger = self.checkpoint_path + 'unet_training_logger_{}i_{}e_{}bs_{}lr_{}.csv'.format(utils.LIMIT_IMAGES_SEGMENTATION_PKL, utils.EPOCHS_UNET, utils.BS_UNET, str(utils.LR_UNET).split(".")[1], utils.FUNCTION_UNET)

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
        print("UNet model and history are saved")

class BemboNET():

    def get_model(self):
        model = tf.keras.Sequential([
            # resize_and_rescale,
            # data_augmentation,

            layers.Conv2D(32, 3, strides=3, padding='same', input_shape=(224, 224, 3)),
            layers.LeakyReLU(alpha=0.3),
            layers.MaxPooling2D(pool_size=(3, 3), strides=3, padding='same'),

            layers.Dense(256, activation=layers.LeakyReLU(alpha=0.3)),
            layers.Conv2D(60, 3, strides=3, padding='same'),
            layers.LeakyReLU(alpha=0.3),
            layers.MaxPooling2D(pool_size=(3, 3), strides=3, padding='same'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),

            layers.Conv2D(60, 2, strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.3),
            layers.MaxPooling2D(pool_size=2, strides=2, padding='same'),
            layers.Dropout(0.2),
            layers.BatchNormalization(),

            layers.Dense(1024),
            layers.LeakyReLU(alpha=0.3),
            layers.Conv2D(90, 3, strides=1, padding='same'),
            layers.LeakyReLU(alpha=0.3),
            layers.MaxPooling2D(pool_size=(3, 3), strides=3, padding='same'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),

            layers.Conv2D(178, 3, strides=3, padding='same'),
            layers.LeakyReLU(alpha=0.3),
            layers.MaxPooling2D(pool_size=(3, 3), strides=3, padding='same'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),

            layers.Dense(1024),
            layers.LeakyReLU(alpha=0.3),
            layers.Conv2D(60, 2, strides=2, padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=2, strides=2, padding='same'),
            layers.Dropout(0.5),
            layers.BatchNormalization(),

            layers.Dense(512, activation='sigmoid'),
            layers.Conv2D(90, 3, strides=3, padding='same', activation='sigmoid'),
            # layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'),
            # layers.Dropout(0.5),
            # layers.BatchNormalization(),

            layers.Flatten(),

            layers.Dense(512, activation='sigmoid'),
            layers.Dense(1, activation="sigmoid")
        ])

        # model.compile(optimizer = tf.optimizers.Adam(),loss = 'binary_crossentropy',metrics=['accuracy'])
        # opt = SGD(lr=0.01)
        opt = tf.keras.optimizers.Adam(lr=0.00001)
        # model.compile(loss = "binary_crossentropy", optimizer=opt, metrics=['accuracy'])
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM),
                      metrics=['accuracy'])

        return model