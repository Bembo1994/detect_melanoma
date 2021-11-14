from dataset import dataset
from machine_learning import neural_network, svm
from utils import utils
from preprocessing import preprocess_image
from keras.models import load_model
from matplotlib import pyplot as plt
from tensorflow.keras.utils import normalize
import os, random
import cv2
import random
import numpy as np
import sys
import argparse

def main():

    ds = dataset.Dataset()

    '''dataset download and analysis'''
    if utils.DOWNLOAD_DATASET:
        ds.download_metadata()
        ds.download_dataset()
        ds.download_mask()

    '''
    for segmentation -> UNet
    from images to pickle, for the training and validation dataset set LIMIT_IMAGES_SEGMENTATION_PKL and for set the test dataset set the TEST_SET_SIZE
    '''
    #ds.dataset_segmentation_to_pickle(0, utils.LIMIT_IMAGES_SEGMENTATION_PKL, False) # for training
    #ds.dataset_segmentation_to_pickle(utils.LIMIT_IMAGES_SEGMENTATION_PKL, utils.TEST_SET_SIZE_SEGMENTATION, True) # for testing

    #training and test unet
    if eval(args.train_unet) or eval(args.test_unet):
        unet = neural_network.UNet()
        unet_model = unet.get_model(utils.IMG_SIZE_UNET, utils.IMG_SIZE_UNET, utils.IMG_CHANNELS_UNET)
        if eval(args.train_unet):
            X_train, X_val, y_train, y_val = ds.get_train_and_val_set_for_segmentation()
            unet.train(unet_model, X_train, X_val, y_train, y_val)
            _, acc = unet_model.evaluate(X_val, y_val)
            print("Accuracy = ", (acc * 100.0), "%")

        if eval(args.test_unet):
            _, X_test, _, y_test = ds.get_test_set_for_segmentation()
            # evaluate model
            _, acc = unet_model.evaluate(X_test, y_test)
            print("TEST Accuracy = ", (acc * 100.0), "%")

    '''for classification -> VGG16'''
    #ds.dataset_classification_to_pickle(False) #make training and validation set

    #ds.dataset_classification_to_pickle(True) #make test set

    # training and test vgg
    if eval(args.train_vgg) or eval(args.test_vgg):
        vgg16 = neural_network.NN_VGG16()
        vgg16_model = vgg16.get_model()

        if eval(args.train_vgg):

            if eval(args.ds_unet):#using datest preprocessed with unet_preprocessing function
                X_train, X_val, y_train, y_val = ds.get_train_and_val_set_for_classification(False, False)
            else: #using datest preprocessed with cv_preprocessing function
                X_train, X_val, y_train, y_val = ds.get_train_and_val_set_for_classification(True, False)

            vgg16.train(vgg16_model, X_train, X_val, y_train, y_val)
            _, acc = vgg16_model.evaluate(X_val, y_val)
            print("Accuracy = ", (acc * 100.0), "%")

        if eval(args.test_vgg):
            if eval(args.ds_unet):#using datest preprocessed with unet_preprocessing function
                _, X_test, _, y_test = ds.get_train_and_val_set_for_classification(False, True)
            else: #using datest preprocessed with cv_preprocessing function
                _, X_test, _, y_test = ds.get_train_and_val_set_for_classification(True, True)

            # evaluate model
            _, acc = vgg16_model.evaluate(X_test, y_test)
            print("TEST Accuracy = ", (acc * 100.0), "%")

if __name__== "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_unet", help="(Boolean) Train unet ?")
    parser.add_argument("--test_unet", help="(Boolean) Test unet ?")
    parser.add_argument("--train_vgg", help="(Boolean) Train vgg16 ?")
    parser.add_argument("--test_vgg", help="(Boolean) Test vgg16 ?")
    parser.add_argument("--ds_unet", help="(Boolean) Train vgg16 with unet preprocessing?")
    args = parser.parse_args()
    main()