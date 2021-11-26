from dataset.API.isic_api import ISICApi
from utils import utils
from collections import defaultdict
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.keras.utils import normalize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocessing import preprocess_image
from keras.models import load_model
from machine_learning import neural_network
import os
import json
import cv2
import pandas as pd
import numpy as np
import random
import sys
import pickle
import shutil
from sklearn.utils import shuffle
from scipy import ndimage


class Dataset():

    def __init__(self):
        self.limit = utils.LIMIT_IMAGES
        self.real_path = os.path.dirname(os.path.realpath(__file__))
        self.savePath = self.real_path+"/ISICArchive/"
        self.segmentation_path = self.savePath + "Segmentation/"
        self.classification_path = self.savePath + "Classification/"
        # Initialize the API; no login is necessary for public data
        self.api = ISICApi()
        self.make_dirs(self.savePath)
        self.make_dirs(self.segmentation_path)
        self.make_dirs(self.classification_path)
        self.image_directory = self.segmentation_path + 'img/'
        self.mask_directory = self.segmentation_path + 'mask/'
        self.make_dirs(self.mask_directory)
        self.make_dirs(self.image_directory)
        self.img_for_segmentation = set()
        self.num_sample_for_segmentation = utils.LIMIT_IMAGES_SEGMENTATION_DOWNLOAD
        self.preprocessor = preprocess_image.Preprocessor()
        self.size_test_set = utils.SIZE_TEST_SET

    def make_dirs(self,path):
        if not os.path.exists(path):
            print("Make directory : {}".format(path))
            os.makedirs(path)

    def download_metadata(self):
        #download and save metadata
        if not os.path.isfile(self.savePath+"metadata.json") :
            imageList = self.api.getJson('image?limit='+str(self.limit)+'&offset=0&sort=name')
            print('Fetching metadata for %s images' % len(imageList))
            imageDetails = []
            for image in imageList:
                # Fetch the full image details
                imageDetail = self.api.getJson('image/%s' % image['_id'])
                imageDetails.append(imageDetail)
            with open(self.savePath+"metadata.json", 'w') as outfile:
                json.dump(imageDetails, outfile)
            print("Saved file : {}".format(self.savePath+"metadata.json"))
        else:
            print("File already saved : {}".format(self.savePath + "metadata.json"))

        if not os.path.isfile(self.savePath+'dataframe.csv'):
            print("Creating csv file")
            with open(self.savePath+"metadata.json") as f:
                df = pd.read_json(f)
            column_to_expand = ["creator", "dataset", "meta", "notes"]
            for column in column_to_expand:
                data_normalize = pd.json_normalize(df[column])
                for c in data_normalize.columns:
                    data_normalize.rename(columns={c: column + "." + c}, inplace=True)
                df.drop(column, axis=1, inplace=True)
                df = pd.concat([df, data_normalize], axis=1, join="inner")
            #69445 rows (images) x 36 columns
            df.to_csv(self.savePath+'dataframe.csv',index=False)

        if not os.path.isfile(self.savePath+'dataframe_cleaned.csv') :
            # cleaning dataframe, get only the nevus and melanoma images
            df = pd.read_csv(self.savePath + 'dataframe.csv')
            df = df[df['meta.clinical.benign_malignant'].notna()]
            df = df.set_index("meta.clinical.benign_malignant")
            df = df.drop(["indeterminate", "indeterminate/benign", "indeterminate/malignant"])
            df = df[df['meta.clinical.diagnosis'].notna()]
            df = df.reset_index()
            df = df.set_index("meta.clinical.diagnosis")
            df = df.drop(
                ["actinic keratosis", "angiofibroma or fibrous papule", "angioma", "atypical melanocytic proliferation",
                 "basal cell carcinoma"])
            df = df.drop(
                ["cafe-au-lait macule", "dermatofibroma", "lentigo NOS", "lentigo simplex", "lichenoid keratosis"])
            df = df.drop(["other", "scar", "seborrheic keratosis", "solar lentigo", "squamous cell carcinoma"])
            # 33468 rows -> images
            df = df.reset_index()
            df = df.set_index("dataset.name")
            df = df.drop(["SONIC"])     #the images of this class are full of artifacts
            df.reset_index()
            # 24217 rows (images) x 35 columns
            df.to_csv(self.savePath+'dataframe_cleaned.csv',index=False)

    def download_dataset(self):

        if os.path.isfile(self.savePath+'dataframe_cleaned.csv'):
            df = pd.read_csv(self.savePath+'dataframe_cleaned.csv')
        else:
            sys.exit("ERROR : Dataframe is not saved")

        print('Checking images')
        for i, name in enumerate(df["name"]):
            name_ext = str(name)+".jpg"
            id = df.iloc[i]["_id"]
            b_or_m = df.iloc[i]["meta.clinical.benign_malignant"]
            path = self.classification_path + b_or_m + "/"
            if len(os.listdir(path)) <= utils.NUMBER_IMG_FOR_CLASS :
                imageFileOutputPath = os.path.join(path, name_ext)
                if not (os.path.isfile((imageFileOutputPath))):
                    imageFileResp = self.api.get('image/%s/download' % id)
                    imageFileResp.raise_for_status()
                    with open(imageFileOutputPath, 'wb') as imageFileOutputStream:
                        for chunk in imageFileResp:
                            imageFileOutputStream.write(chunk)
        print("Downloaded dataset")

    def download_mask(self):
        if not (os.path.isfile(self.savePath+'dataframe.csv') or os.path.isfile(self.savePath+'dataframe_cleaned.csv')):
            sys.exit("ERROR : Before download the dataset")

        df1 = pd.read_csv(self.savePath+'dataframe.csv')
        df2 = pd.read_csv(self.savePath + 'dataframe_cleaned.csv')
        df1.set_index('_id', inplace=True)
        df2.set_index('_id', inplace=True)
        df = df1.drop(df2.index) #get only images not used for classification

        print("{} possible images to download, but only {} images are downloading for segmentation".format(len(df), self.num_sample_for_segmentation))
        #get images of dataset named "SONIC" because force the network to not overfitting
        for i, name in enumerate(df["dataset.name"]):
            if name == "SONIC" and len(self.img_for_segmentation)<self.num_sample_for_segmentation:
                self.img_for_segmentation.add(df["_id"][i])

        #check if need to download more images
        if len(os.listdir(self.mask_directory)) < self.num_sample_for_segmentation and len(os.listdir(self.mask_directory)) == len(os.listdir(self.image_directory)):
            for i, img_id in enumerate(self.img_for_segmentation) :
                name = df.iloc[list(df["_id"]).index(img_id)]["name"]
                flag = False
                if not os.path.isfile(self.mask_directory+"%s.tiff" % name) and not os.path.isfile(self.image_directory+"%s.tiff" % name):
                    flag = True
                if flag :
                    segmentation = self.api.getJson('segmentation?imageId=' + img_id)
                    if len(segmentation)>0: #check if image mask exists
                        imageFileResp_img = self.api.get('image/%s/download' % img_id)

                        imageFileResp_seg = self.api.get('segmentation/%s/mask' % segmentation[0]['_id'])

                        imageFileResp_seg.raise_for_status()
                        imageFileResp_img.raise_for_status()

                        imageFileOutputPath_seg = os.path.join(self.segmentation_path+"mask/", '%s.tiff' % name)
                        imageFileOutputPath_img = os.path.join(self.segmentation_path+"img/", '%s.tiff' % name)

                        with open(imageFileOutputPath_seg, 'wb') as imageFileOutputStream_seg:
                            for chunk in imageFileResp_seg:
                                imageFileOutputStream_seg.write(chunk)
                        with open(imageFileOutputPath_img, 'wb') as imageFileOutputStream_img:
                            for chunk in imageFileResp_img:
                                imageFileOutputStream_img.write(chunk)

            print("Downloaded image for segmentation")

    def data_augmentation(self):
        if len(os.listdir(self.classification_path+'benign/')) > len(os.listdir(self.classification_path+'malignant/')):
            imgs_to_aug = os.listdir(self.classification_path+'malignant/')
            for i in imgs_to_aug:
                image = cv2.imread(self.classification_path+'malignant/' + i)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                rotated = ndimage.rotate(image, 180)
                # Filename
                filename = self.classification_path+'malignant/' + str(i.split(".")[0]) + "_aug180." + str(i.split(".")[1])
                cv2.imwrite(filename, rotated)
        else:
            print("Data augmentation already made")

    def get_dataset_segmentation(self, is_for_test_set):

        dataset = []
        label = []

        images = os.listdir(self.image_directory)
        masks = os.listdir(self.mask_directory)
        size = utils.IMG_SIZE_UNET

        if is_for_test_set :
            images = images[-self.size_test_set:]
            masks = masks[-self.size_test_set:]
        else :
            images = images[:-self.size_test_set]
            masks = masks[:-self.size_test_set]

        for image_name in images:
            image = cv2.imread(self.image_directory + image_name, 0)
            image = Image.fromarray(image)
            image = image.resize((size, size))
            dataset.append(np.array(image))

        for image_name in masks:
            image = cv2.imread(self.mask_directory + image_name, 0)
            image = Image.fromarray(image)
            image = image.resize((size, size))
            label.append(np.array(image))

        dataset, label = shuffle(dataset, label, random_state=71)
        dataset = np.array(dataset)
        label = np.array(label)
        # Normalize images
        dataset = np.expand_dims(normalize(np.array(dataset), axis=1), 3)
        # D not normalize masks, just rescale to 0 to 1.
        label = np.expand_dims((np.array(label)), 3) / 255.

        X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.1, random_state=123)

        return X_train, X_test, y_train, y_test

    def get_dataset_classification(self, cv_or_unet_preprocessing, is_for_test_set):

        dataset = []
        label = []

        benigns = os.listdir(self.classification_path + "benign/")
        malignants = os.listdir(self.classification_path + "malignant/")

        if is_for_test_set :
            benigns = benigns[-self.size_test_set:]
            malignants = malignants[-self.size_test_set:]
        else :
            benigns = benigns[:-self.size_test_set]
            malignants = malignants[:-self.size_test_set]

        for image_name in benigns:
            if cv_or_unet_preprocessing == "cv":
                result = self.preprocessor.cv_preprocessing(self.classification_path + "benign/" + image_name)
            else :
                result = self.preprocessor.unet_preprocessing(self.classification_path + "benign/" + image_name)
            dataset.append(np.array(result))
            label.append(0)

        for image_name in malignants:
            if cv_or_unet_preprocessing == "cv":
                result = self.preprocessor.cv_preprocessing(self.classification_path + "malignant/" + image_name)
            else :
                result = self.preprocessor.unet_preprocessing(self.classification_path + "malignant/" + image_name)
            dataset.append(np.array(result))
            label.append(1)

        dataset, label = shuffle(dataset, label, random_state=71)
        dataset = np.array(dataset)
        label = np.array(label)


        X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.1, random_state=123)

        return X_train, X_test, y_train, y_test