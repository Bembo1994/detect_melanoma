from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.client import device_lib
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten
from pathlib import Path
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalMaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, GlobalAveragePooling2D, Activation
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger
import os
import tensorflow as tf
import numpy as np
from utils import utils
import pandas as pd
from keras.utils.vis_utils import plot_model


class NeuralNetworks():

    def __init__(self, network, cv_or_unet_preprocessing):

        self.cv_or_unet_preprocessing = cv_or_unet_preprocessing
        self.head_name = network
        self.path_model_tmp = ""
        if self.head_name == "unet":
            self.img_size = utils.IMG_SIZE_UNET
            self.img_channels = utils.IMG_CHANNELS_UNET
            self.epochs = utils.EPOCHS_UNET
            self.batch_size = utils.BS_UNET
            self.learning_rate = utils.LR_UNET
            self.activation_function = utils.FUNCTION_UNET
        if self.head_name == "vgg16":
            self.img_size = utils.IMG_SIZE_VGG
            self.epochs = utils.EPOCHS_VGG
            self.batch_size = utils.BS_VGG
            self.learning_rate = utils.LR_VGG
            self.activation_function = utils.FUNCTION_VGG
        if self.head_name == "resnet":
            self.img_size = utils.IMG_SIZE_RESNET
            self.epochs = utils.EPOCHS_RESNET
            self.batch_size = utils.BS_RESNET
            self.learning_rate = utils.LR_RESNET
            self.activation_function = utils.FUNCTION_RESNET
        if self.head_name == "inception_v3" :
            self.img_size = utils.IMG_SIZE_INCEPTION
            self.epochs = utils.EPOCHS_INCEPTION
            self.batch_size = utils.BS_INCEPTION
            self.learning_rate = utils.LR_INCEPTION
            self.activation_function = utils.FUNCTION_INCEPTION

        self.head_network = ""
        self.checkpoint_path = os.path.dirname(os.path.realpath(__file__)) + "/checkpoints/"
        self.make_dirs(self.checkpoint_path)

    def make_dirs(self, path):
        if not os.path.exists(path):
            print("Make directory : {}".format(path))
            os.makedirs(path)

    def get_model(self):
        if self.head_name == "unet":
            return self.get_unet_model()
        if self.head_name == "vgg16":
            return self.get_vgg_model()
        if self.head_name == "resnet":
            return self.get_resnet_model()
        if self.head_name == "inception_v3" :
            return self.get_inception_v3_model()

    def check_if_already_train(self):
        checkpoints_dirs = os.listdir(self.checkpoint_path)
        for dir in checkpoints_dirs:
            if self.head_name in dir and "unet" in self.head_name:
                checkpoint_path = self.checkpoint_path + "{}_{}_lr_{}_bs_{}_{}".format(self.head_name,
                                                                                                   self.activation_function,
                                                                                                   str(self.learning_rate).replace(".","-"),
                                                                                                   self.batch_size)
                print("Loading model")
                return True, checkpoint_path
            elif self.head_name in dir and "unfrozen" in dir:
                checkpoint_path = self.checkpoint_path + "{}_{}_{}_lr_{}_bs_{}_{}_{}".format(self.head_name,
                                                                                                   self.activation_function,
                                                                                                   self.cv_or_unet_preprocessing,
                                                                                                   str(utils.LR_UNFROZEN).replace(".","-"),
                                                                                                   self.batch_size,
                                                                                                   "unfrozen", self.head_network)
                print("Loading model")
                return True, checkpoint_path
        return False, ""

    def get_unet_model(self):
        self.head_network = "_rmsprop"
        flag, checkpoint_path = self.check_if_already_train()
        if flag:
            return load_model(checkpoint_path)

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
        self.head_network = "2x4096"
        flag, checkpoint_path = self.check_if_already_train()
        if flag:
            return load_model(checkpoint_path)

        base_model = VGG16(weights="imagenet", include_top=False, input_shape=(self.img_size, self.img_size, 3))
        
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(4096)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(4096)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.5)(x)
        #x = Dense(128)(x)
        #x = BatchNormalization()(x)
        #x = Activation("relu")(x)
        #x = Dropout(0.5)(x)
        prediction = Dense(1, activation=self.activation_function)(x)
        model = Model(inputs=base_model.input, outputs=prediction)
        adam = Adam(lr=self.learning_rate, decay=0.0, epsilon=None)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def get_resnet_model(self):
        self.head_network = "3x128"
        flag, checkpoint_path = self.check_if_already_train()
        if flag:
            return load_model(checkpoint_path)
        base_model = ResNet50(weights='imagenet', include_top=False,  input_tensor=Input(shape=(self.img_size,self.img_size,3)))

        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.5)(x)
        prediction = Dense(1, activation=self.activation_function)(x)
        model = Model(inputs=base_model.input, outputs=prediction)
        adam = Adam(lr=self.learning_rate, decay=0.0, epsilon=None)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def get_inception_v3_model(self):
        self.head_network = "4x128_LOAD_CORRECT______2"
        flag, checkpoint_path = self.check_if_already_train()
        if flag:
            model = load_model(checkpoint_path)
            plot_model(model, to_file='model_plot.png', show_shapes=False,
                       show_layer_names=True)

            return model

        base_model = tf.keras.applications.InceptionV3(weights='imagenet', input_shape=(self.img_size, self.img_size, 3), include_top=False)
        base_model.trainable = False
        #inputs = keras.Input(shape=(self.img_size, self.img_size, 3))
        #x = base_model(inputs, training=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        prediction = Dense(1, activation=self.activation_function)(x)
        model = Model(inputs=base_model.input, outputs=prediction)
        adam = Adam(lr=self.learning_rate, decay=0.0, epsilon=None)
        model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy"])
        plot_model(model, to_file='/home/bembo/Scrivania/detect_melanoma/model_plot.png', show_shapes=True, show_layer_names=False)
        return model
        
    def scheduler(self, epoch, lr):
        with open(self.path_model_tmp+self.head_name+".txt", "a") as f:
            f.write("{}\t{}\n".format(epoch,lr))
        return lr

    def fit_model(self, model, X_train, X_test, y_train, y_test, csv_logger, model_checkpoint_callback):
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.1, patience=3, verbose=0, mode="max")
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, mode="max", restore_best_weights=True)
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
        model.fit(X_train, y_train,
                  batch_size=self.batch_size,
                  verbose=1,
                  epochs=self.epochs,
                  validation_data=(X_test, y_test),
                  shuffle=False,
                  callbacks=[csv_logger, model_checkpoint_callback, reduce_lr, early_stopping, lr_scheduler])

    def train(self, model, X_train, X_test, y_train, y_test, is_all_unfrozen):
        lr = self.learning_rate
        if is_all_unfrozen:
            model.trainable = True
            lr = utils.LR_UNFROZEN
            opt = Adam(learning_rate=lr)  # 0.001 default
            model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
            is_frozen = "unfrozen"
        else :
            is_frozen = "frozen"
        if self.head_name != "unet":
            checkpoint_path = self.checkpoint_path + "{}_{}_{}_lr_{}_bs_{}_{}_{}/".format(self.head_name, self.activation_function, self.cv_or_unet_preprocessing, str(lr).replace(".","-"), self.batch_size, is_frozen, self.head_network)
        else:
            checkpoint_path = self.checkpoint_path + "{}_{}_{}_lr_{}_bs_{}/".format(self.head_name, self.activation_function, self.cv_or_unet_preprocessing, str(lr).replace(".","-"), self.batch_size)
        self.make_dirs(checkpoint_path)
        print(model.summary())
        self.path_model_tmp = checkpoint_path
        
        class ModelCheckpoint_tweaked(tf.keras.callbacks.ModelCheckpoint):
            def __init__(self,
                   filepath,
                   monitor='val_loss',
                   verbose=0,
                   save_best_only=False,
                   save_weights_only=False,
                   mode='auto',
                   save_freq='epoch',
                   options=None,
                   **kwargs):
        
                super(ModelCheckpoint_tweaked, self).__init__(filepath,
                   monitor,
                   verbose,
                   save_best_only,
                   save_weights_only,
                   mode,
                   save_freq,
                   options,
                   **kwargs)
        
        csv_logger = CSVLogger(checkpoint_path+"logger.csv", append=True)
        
        cb_model_checkpoint = ModelCheckpoint_tweaked(checkpoint_path,
                                              monitor='val_accuracy',
                                              save_best_only=True,
                                              mode='max',
                                              verbose=1)
        if tf.test.is_gpu_available():
            with tf.device('/device:GPU:0'):
                print("Work with GPU")
                self.fit_model(model, X_train, X_test, y_train, y_test, csv_logger, cb_model_checkpoint)
        else:
            print("Work with CPU")
            self.fit_model(model, X_train, X_test, y_train, y_test, csv_logger, cb_model_checkpoint)