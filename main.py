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


def test_model(model, test_ds):
    _, acc = model.evaluate(test_ds)
    print("Accuracy (TEST) = ", (acc * 100.0), "%")

def train_model(net, model, train_ds, val_ds, is_fine_tuning):
    net.train(model, train_ds, val_ds, is_fine_tuning)
    _, acc = model.evaluate(val_ds)
    print("Accuracy (VALIDATION) = ", (acc * 100.0), "%")

def main():
    cv_or_unet_preprocessing = args.cv_or_unet_preprocessing
    ds = dataset.Dataset()
    '''dataset download and analysis'''
    #ds.download_metadata()
    #ds.download_dataset_classification()
    #ds.download_dataset_segmentation()
    ds.data_augmentation()

    ''' for segmentation -> UNet '''
    #training and test unet
    if eval(args.train_unet) or eval(args.test_unet):
        unet = neural_network.NeuralNetworks(True, False, False, "")
        unet_model = unet.get_unet_model()
        if eval(args.train_unet):
            X_train, X_val, y_train, y_val = ds.get_dataset_segmentation(False)
            train_model(unet, unet_model, X_train, X_val, y_train, y_val, True)

        if eval(args.test_unet):
            _, X_test, _, y_test = ds.get_dataset_segmentation(True)
            test_model(unet_model, X_test, y_test)

    '''for classification -> VGG16'''
    # training and test vgg
    if eval(args.train_vgg) or eval(args.test_vgg):
        vgg16 = neural_network.NeuralNetworks(False, True, False, False, cv_or_unet_preprocessing)
        vgg16_model = vgg16.get_vgg_model()
        if eval(args.train_vgg):
            train_ds, val_ds = ds.get_dataset_classification_test("vgg16")
            print("Transfer Learning")
            train_model(vgg16, vgg16_model, train_ds, val_ds, False)
        if eval(args.test_vgg):
            test_ds = ds.get_dataset_classification_test("vgg16")
            test_model(vgg16_model, test_ds)

    '''for classification -> ResNet50'''
    # training and test vgg
    if eval(args.train_resnet) or eval(args.test_resnet):
        resnet = neural_network.NeuralNetworks(False, False, True, False, cv_or_unet_preprocessing)
        resnet_model = resnet.get_resnet_model()
        if eval(args.train_resnet):
            train_ds, val_ds = ds.get_dataset_classification_train("resnet")
            #X_train, X_test, y_train, y_test = ds.get_dataset_classification(cv_or_unet_preprocessing, False, "resnet")
            print("Feature Extraction")
            train_model(resnet, resnet_model, train_ds, val_ds, False)
        if eval(args.test_resnet):
            test_ds = ds.get_dataset_classification_test("resnet")
            test_model(resnet_model, test_ds)

    '''for classification -> Xception'''
    # training and test vgg
    if eval(args.train_xception) or eval(args.test_xceptiont):
        Xception = neural_network.NeuralNetworks(False, False, False, True, cv_or_unet_preprocessing)
        Xception_model = Xception.get_resnet_model()
        if eval(args.train_xception):
            train_ds, val_ds = ds.get_dataset_classification_train("xception")
            #X_train, X_test, y_train, y_test = ds.get_dataset_classification(cv_or_unet_preprocessing, False, "xception")
            print("TransferLearning")
            train_model(Xception, Xception_model, train_ds, val_ds, False)
            #train_model(Xception, Xception_model, X_train, X_test, y_train, y_test, False)
        if eval(args.test_xceptiont):
            #X_test, y_test = ds.get_dataset_classification(cv_or_unet_preprocessing, True, "xception")
            test_ds = ds.get_dataset_classification_test("xception")
            test_model(Xception_model, test_ds)

if __name__== "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_unet", help="(Boolean) Train UNet ?")
    parser.add_argument("--test_unet", help="(Boolean) Test UNet ?")
    parser.add_argument("--train_vgg", help="(Boolean) Train VGG16 ?")
    parser.add_argument("--test_vgg", help="(Boolean) Test VGG16 ?")
    parser.add_argument("--train_resnet", help="(Boolean) Train ResNet50 ?")
    parser.add_argument("--test_resnet", help="(Boolean) Test ResNet50 ?")
    parser.add_argument("--train_xception", help="(Boolean) Train Xception ?")
    parser.add_argument("--test_xceptiont", help="(Boolean) Test Xception ?")
    parser.add_argument("--cv_or_unet_preprocessing", help="(type in 'cv' or 'unet') Train neural networks with cv or unet preprocessing?")
    args = parser.parse_args()
    main()