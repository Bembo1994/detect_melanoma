from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten
from pathlib import Path
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalMaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger
import os
import tensorflow as tf
import numpy as np
from utils import utils
import pandas as pd

class NeuralNetworks():

    def __init__(self, is_unet, is_vgg, is_resnet, cv_or_unet_preprocessing):

        self.cv_or_unet_preprocessing = cv_or_unet_preprocessing

        if is_unet:
            self.img_size = utils.IMG_SIZE_UNET
            self.img_channels = utils.IMG_CHANNELS_UNET
            self.checkpoint_path = os.path.dirname(os.path.realpath(__file__)) + "/checkpoints/unet/"
            self.make_dirs(self.checkpoint_path)
            self.epochs = utils.EPOCHS_UNET
            self.batch_size = utils.BS_UNET
            self.learning_rate = utils.LR_UNET
            self.activation_function = utils.FUNCTION_UNET
            self.head_name = "unet"
        if is_vgg:
            self.img_size = utils.IMG_SIZE_VGG
            self.checkpoint_path = os.path.dirname(os.path.realpath(__file__)) + "/checkpoints/vgg16/"
            self.make_dirs(self.checkpoint_path)
            self.epochs = utils.EPOCHS_VGG
            self.batch_size = utils.BS_VGG
            self.learning_rate = utils.LR_VGG
            self.activation_function = utils.FUNCTION_VGG
            self.head_name = "vgg"
        if is_resnet:
            self.img_size = utils.IMG_SIZE_RESNET
            self.checkpoint_path = os.path.dirname(os.path.realpath(__file__)) + "/checkpoints/resnet50/"
            self.make_dirs(self.checkpoint_path)
            self.epochs = utils.EPOCHS_RESNET
            self.batch_size = utils.BS_RESNET
            self.learning_rate = utils.LR_RESNET
            self.activation_function = utils.FUNCTION_RESNET
            self.head_name = "resnet"
        if is_bembonet:
            self.img_size = utils.IMG_SIZE_BEMBONET
            self.checkpoint_path = os.path.dirname(os.path.realpath(__file__)) + "/checkpoints/resnet50/"
            self.make_dirs(self.checkpoint_path)
            self.epochs = utils.EPOCHS_BEMBONET
            self.batch_size = utils.BS_BEMBONET
            self.learning_rate = utils.LR_BEMBONET
            self.activation_function = utils.FUNCTION_BEMBONET
            self.head_name = "bembonet"

        app_name_model = '{}_model_{}_{}e_{}bs_{}lr_{}.hdf5'.format(self.head_name, self.cv_or_unet_preprocessing, self.epochs,
                                                                       self.batch_size,
                                                                       str(self.learning_rate).split(".")[1],
                                                                       self.activation_function)
        app_name_logger = '{}_training_logger_{}_{}e_{}bs_{}lr_{}.csv'.format(self.head_name, self.cv_or_unet_preprocessing, self.epochs,
                                                                       self.batch_size,
                                                                       str(self.learning_rate).split(".")[1],
                                                                       self.activation_function)

        self.name_model = self.checkpoint_path + app_name_model
        self.name_logger = self.checkpoint_path + app_name_logger

    def make_dirs(self, path):
        if not os.path.exists(path):
            print("Make directory : {}".format(path))
            os.makedirs(path)

    def get_unet_model(self):

        if os.path.isfile(self.name_model) and os.path.isfile(self.name_logger) :
            print("UNet is already train")
            return load_model(self.name_model)

        # Build the model
        inputs = Input((self.img_size, self.img_size, self.img_channels))
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
        opt = Adam(learning_rate=self.learning_rate) # 0.001 default
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return model

    def get_vgg_model(self):

        if os.path.isfile(self.name_model) and os.path.isfile(self.name_logger) :
            print("VGG16 is already train")
            return load_model(self.name_model)

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
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        #model.add(Dense(4096, activation='softmax'))
        model.add(Dense(1, activation='sigmoid'))  # sigmoid or relu

        # Show a summary of the model. Check the number of trainable parameters
        print(model.summary())

        opt = Adam(learning_rate=self.learning_rate)  # 0.001 default
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        print(model.summary())

        return model

    def get_resnet_model(self):

        if os.path.isfile(self.name_model) and os.path.isfile(self.name_logger) :
            print("ResNet50 is already train")
            return load_model(self.name_model)

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
        adam = Adam(lr=self.learning_rate)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def get_bembonet_model(self):

        if os.path.isfile(self.name_model) and os.path.isfile(self.name_logger) :
            print("BemboNet is already train")
            return load_model(self.name_model)

        model = tf.keras.Sequential()
        model.add.Conv2D(32, 3, strides=3, padding='same', input_shape=(224, 224, 3)),
        model.add.LeakyReLU(alpha=0.3),
        model.add.MaxPooling2D(pool_size=(3, 3), strides=3, padding='same'),

        model.add.Dense(256, activation=layers.LeakyReLU(alpha=0.3)),
        model.add.Conv2D(60, 3, strides=3, padding='same'),
        model.add.LeakyReLU(alpha=0.3),
        model.add.MaxPooling2D(pool_size=(3, 3), strides=3, padding='same'),
        model.add.Dropout(0.3),
        model.add.BatchNormalization(),

        model.add.Conv2D(60, 2, strides=2, padding='same'),
        model.add.LeakyReLU(alpha=0.3),
        model.add.MaxPooling2D(pool_size=2, strides=2, padding='same'),
        model.add.Dropout(0.2),
        model.add.BatchNormalization(),

        model.add.Dense(1024),
        model.add.LeakyReLU(alpha=0.3),
        model.add.Conv2D(90, 3, strides=1, padding='same'),
        model.add.LeakyReLU(alpha=0.3),
        model.add.MaxPooling2D(pool_size=(3, 3), strides=3, padding='same'),
        model.add.Dropout(0.3),
        model.add.BatchNormalization(),

        model.add.Conv2D(178, 3, strides=3, padding='same'),
        model.add.LeakyReLU(alpha=0.3),
        model.add.MaxPooling2D(pool_size=(3, 3), strides=3, padding='same'),
        model.add.Dropout(0.3),
        model.add.BatchNormalization(),

        model.add.Dense(1024),
        model.add.LeakyReLU(alpha=0.3),
        model.add.Conv2D(60, 2, strides=2, padding='same', activation='relu'),
        model.add.MaxPooling2D(pool_size=2, strides=2, padding='same'),
        model.add.Dropout(0.5),
        model.add.BatchNormalization(),

        model.add.Dense(512, activation='sigmoid'),
        model.add.Conv2D(90, 3, strides=3, padding='same', activation='sigmoid'),
        # layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'),
        # layers.Dropout(0.5),
        # layers.BatchNormalization(),

        model.add.Flatten(),

        model.add.Dense(512, activation='sigmoid'),
        model.add.Dense(1, activation="sigmoid")


        # model.compile(optimizer = tf.optimizers.Adam(),loss = 'binary_crossentropy',metrics=['accuracy'])
        # opt = SGD(lr=0.01)
        opt = tf.keras.optimizers.Adam(self.learning_rate)
        # model.compile(loss = "binary_crossentropy", optimizer=opt, metrics=['accuracy'])
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM),
                      metrics=['accuracy'])

        return model

    def fit_model(self, model,  X_train, X_val, y_train, y_val, csv_logger, model_checkpoint_callback):
        #weights = compute_class_weight('balanced', np.unique(y_train), y_train)
        #w = {0: weights[0], 1:weights[1]}
        #print(w)
        #dimezza learning reate ogni 3 epoche se val_accuracy non incrementa
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3, verbose=0, mode="max")
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, mode="max", restore_best_weights=True)

        model.fit(X_train, y_train,
                  batch_size=self.batch_size,
                  verbose=1,
                  epochs=self.epochs,
                  validation_data=(X_val, y_val),
                  shuffle=False,
                  #class_weight=w,
                  callbacks=[csv_logger, model_checkpoint_callback, reduce_lr, early_stopping])

    def train(self, model,  X_train, X_val, y_train, y_val, is_fine_tuning):
        if is_fine_tuning:
            name_model = self.name_model.split(".")[0]+"FineTuning"+self.name_model.split(".")[1]
            name_logger = self.name_logger.split(".")[0]+"FineTuning"+self.name_logger.split(".")[1]
        else :
            name_model = self.name_model.split(".")[0]+"FeatureExtraction"+self.name_model.split(".")[1]
            name_logger = self.name_logger.split(".")[0]+"FeatureExtraction"+self.name_logger.split(".")[1]

        if os.path.isfile(name_model) and os.path.isfile(name_logger):
            print("The model : {}\n is already train with this params".format(name_model))
            return

        csv_logger = CSVLogger(self.name_logger)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        if tf.test.is_gpu_available():
            with tf.device('/device:GPU:0'):
                print("Work with GPU")
                self.fit_model(model,  X_train, X_val, y_train, y_val, csv_logger, model_checkpoint_callback)
        else:
            print("Work with CPU")
            self.fit_model(model,  X_train, X_val, y_train, y_val, csv_logger, model_checkpoint_callback)

        model.save(name_model)
        print("Model {}\ncsv logger {} are saved".format(name_model, name_logger))
