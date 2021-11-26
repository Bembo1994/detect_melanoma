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
from tensorflow.keras.callbacks import CSVLogger

def main():

    ds = dataset.Dataset()
    '''dataset download and analysis'''
    #ds.download_metadata()
    #ds.download_dataset()
    #ds.download_mask()
    #ds.data_augmentation()

    ''' for segmentation -> UNet '''

    #training and test unet
    if eval(args.train_unet) or eval(args.test_unet):
        unet = neural_network.UNet()
        unet_model = unet.get_model(utils.IMG_SIZE_UNET, utils.IMG_SIZE_UNET, utils.IMG_CHANNELS_UNET)
        if eval(args.train_unet):
            X_train, X_val, y_train, y_val = ds.get_dataset_segmentation(False)
            unet.train(unet_model, X_train, X_val, y_train, y_val)
            _, acc = unet_model.evaluate(X_val, y_val)
            print("Accuracy = ", (acc * 100.0), "%")

        if eval(args.test_unet):
            _, X_test, _, y_test = ds.get_dataset_segmentation(True)
            # evaluate model
            _, acc = unet_model.evaluate(X_test, y_test)
            print("TEST Accuracy = ", (acc * 100.0), "%")

    '''for classification -> VGG16'''
    # training and test vgg
    if eval(args.train_vgg) or eval(args.test_vgg):
        vgg16 = neural_network.NN_VGG16()
        vgg16_model = vgg16.get_model(args.cv_or_unet_preprocessing)
        if eval(args.train_vgg):
            X_train, X_val, y_train, y_val = ds.get_dataset_classification(args.cv_or_unet_preprocessing, False)
            vgg16.train(vgg16_model, X_train, X_val, y_train, y_val, args.cv_or_unet_preprocessing)
            _, acc = vgg16_model.evaluate(X_test, y_test)
            print("Accuracy (VALIDATION) = ", (acc * 100.0), "%")
        if eval(args.test_vgg):
            _, X_test, _, y_test = ds.get_dataset_classification(args.cv_or_unet_preprocessing, True)
            _, acc = vgg16_model.evaluate(X_test, y_test)
            print("Accuracy (TEST) = ", (acc * 100.0), "%")


    '''for classification -> ResNet50'''
    # training and test vgg
    if eval(args.train_resnet) or eval(args.test_resnet):
        resnet = neural_network.NN_ResNet50()
        resnet_model = resnet.get_model(args.cv_or_unet_preprocessing)
        if eval(args.train_resnet):
            X_train, X_val, y_train, y_val = ds.get_dataset_classification(args.cv_or_unet_preprocessing, False)
            resnet.train(resnet_model, X_train, X_val, y_train, y_val, args.cv_or_unet_preprocessing)
            _, acc = resnet_model.evaluate(X_test, y_test)
            print("Accuracy (VALIDATION) = ", (acc * 100.0), "%")
        if eval(args.test_resnet):
            _, X_test, _, y_test = ds.get_dataset_classification(args.cv_or_unet_preprocessing, True)
            _, acc = resnet_model.evaluate(X_test, y_test)
            print("Accuracy (TEST) = ", (acc * 100.0), "%")

if __name__== "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_unet", help="(Boolean) Train UNet ?")
    parser.add_argument("--test_unet", help="(Boolean) Test UNet ?")
    parser.add_argument("--train_vgg", help="(Boolean) Train VGG16 ?")
    parser.add_argument("--test_vgg", help="(Boolean) Test VGG16 ?")
    parser.add_argument("--train_resnet", help="(Boolean) Train ResNet50 ?")
    parser.add_argument("--test_resnet", help="(Boolean) Test ResNet50 ?")
    parser.add_argument("--cv_or_unet_preprocessing", help="(type in 'cv' or 'unet') Train vgg16 with cv or unet preprocessing?")
    args = parser.parse_args()
    main()