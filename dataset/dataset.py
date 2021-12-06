from dataset.API.isic_api import ISICApi
from utils import utils
from collections import defaultdict
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.keras.utils import normalize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocessing import preprocess_image
from tensorflow.keras.models import load_model
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
import itertools

class Dataset():

    def __init__(self):
        #self.limit = utils.LIMIT_IMAGES
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

        #self.img_for_segmentation = set()
        #self.num_sample_for_segmentation = utils.LIMIT_IMAGES_SEGMENTATION_DOWNLOAD
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

        if not (os.path.isfile(self.savePath+'dataframe_cleaned_classification.csv') or os.path.isfile(self.savePath+'dataframe_cleaned_segmentation.csv')):
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
            tmp = df.reset_index()
            df_c = tmp.set_index("dataset.name")
            df_c = df_c.drop(["SONIC"])     #the images of this class are full of artifacts
            df_c.reset_index()

            ds_name = list(df["dataset.name"].unique())
            drop_names = list(set(ds_name) - set(["SONIC"]))
            df_s = tmp.set_index("dataset.name")
            df_s = df_s.drop(drop_names)
            df_s = df_s.reset_index()

            df_c.to_csv(self.savePath + 'dataframe_cleaned_classification.csv',index=False)
            df_s.to_csv(self.savePath + 'dataframe_cleaned_segmentation.csv', index=False)

    #METTI CONTROLLO PRIMA DI VEDERE SE ESISTE IL FILE
    def download_dataset_classification(self):

        if os.path.isfile(self.savePath+'dataframe_cleaned_classification.csv'):
            df = pd.read_csv(self.savePath+'dataframe_cleaned_classification.csv')
        else:
            sys.exit("ERROR : Dataframe is not saved")

        #df = df.set_index("dataset.name")
        #df = df.drop(["SONIC"])     #the images of this class are full of artifacts, good for segmentation
        #df = df.reset_index()

        for n in df["dataset.name"].unique():
            self.make_dirs(self.classification_path + n +"/benign")
            self.make_dirs(self.classification_path + n + "/malignant")
        print('Checking images')
        for i, name in enumerate(df["name"]):
            if i%1000 == 0 and i > 0:
                print("Images download {}".format(i))
            name_ext = str(name)+".jpg"
            id = df.iloc[i]["_id"]
            ds_name = df.iloc[i]["dataset.name"]
            b_or_m = df.iloc[i]["meta.clinical.benign_malignant"]
            path = self.classification_path + ds_name + "/" + b_or_m + "/"
            #if len(os.listdir(path)) <= utils.NUMBER_IMG_FOR_CLASS :
            imageFileOutputPath = os.path.join(path, name_ext)
            if not (os.path.isfile((imageFileOutputPath))):
                imageFileResp = self.api.get('image/%s/download' % id)
                imageFileResp.raise_for_status()
                with open(imageFileOutputPath, 'wb') as imageFileOutputStream:
                    for chunk in imageFileResp:
                        imageFileOutputStream.write(chunk)
        print("Downloaded dataset")

    def download_dataset_segmentation(self):
        if not os.path.isfile(self.savePath+'dataframe_cleaned_segmentation.csv') :
            sys.exit("ERROR : Before download the dataset")
        df = pd.read_csv(self.savePath+'dataframe_cleaned_segmentation.csv')
        for i, img_id in enumerate(df["_id"]) :
            name = df.iloc[i]["name"]
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
        data_to_aug = os.listdir(self.classification_path+"train/malignant/")
        for i in data_to_aug:
            img = self.preprocessor.read_in_rgb(self.classification_path+"train/malignant/"+i)
            degrees = [90, 180, 270]
            darkness = [20, 35, 50, 60]
            flippings = [1, 0, -1]
            combinations = list(itertools.product(degrees, darkness, flippings))
            random.shuffle(combinations)
            for _ in range(4):
                c = random.choice(combinations)
                # grab the dimensions of the image and calculate the center of the
                # image
                (h, w) = img.shape[:2]
                (cX, cY) = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D((cX, cY), c[0], 1.0)
                rotated = cv2.warpAffine(img, M, (w, h))
                adjusted = cv2.convertScaleAbs(rotated, alpha=1, beta=c[1])
                flipped = cv2.flip(adjusted, c[2])
                combinations.remove(c)
                name = "train/malignant/rotation_{}_dark_{}_flip_{}".format(c[0], c[1], c[2])
                cv2.imwrite(self.classification_path+name+i, cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB))

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
        flag_validation = False
        if is_for_test_set :
            path = self.classification_path + "test"
        else:
            path = self.classification_path + "train"
            flag_validation = True

        benigns = os.listdir(path + "/benign/")
        malignants = os.listdir(path + "/malignant/")

        for image_name in benigns:
            if cv_or_unet_preprocessing == "cv":
                result = self.preprocessor.cv_preprocessing(path + "/benign/" + image_name)
            else :
                result = self.preprocessor.unet_preprocessing(path + "/benign/" + image_name)
            dataset.append(np.array(result))
            label.append(0)

        for image_name in malignants:
            if cv_or_unet_preprocessing == "cv":
                result = self.preprocessor.cv_preprocessing(path + "/malignant/" + image_name)
            else :
                result = self.preprocessor.unet_preprocessing(path + "/malignant/" + image_name)
            dataset.append(np.array(result))
            label.append(1)

        dataset, label = shuffle(dataset, label, random_state=71)
        dataset, label = shuffle(dataset, label, random_state=72)
        dataset, label = shuffle(dataset, label, random_state=73)
        print("BENIGN IMAGES : {}".format(list(label).count(0)))
        print("MALIGNANT IMAGES : {}".format(list(label).count(1)))
        dataset = np.array(dataset)
        label = np.array(label)

        if flag_validation:
            dataset_val = []
            label_val = []
            path = self.classification_path + "validation"
            benigns = os.listdir(path + "/benign/")
            malignants = os.listdir(path + "/malignant/")

            for image_name in benigns:
                if cv_or_unet_preprocessing == "cv":
                    result = self.preprocessor.cv_preprocessing(path + "/benign/" + image_name)
                else:
                    result = self.preprocessor.unet_preprocessing(path + "/benign/" + image_name)
                dataset_val.append(np.array(result))
                label_val.append(0)

            for image_name in malignants:
                if cv_or_unet_preprocessing == "cv":
                    result = self.preprocessor.cv_preprocessing(path + "/malignant/" + image_name)
                else:
                    result = self.preprocessor.unet_preprocessing(path + "/malignant/" + image_name)
                dataset_val.append(np.array(result))
                label_val.append(1)

            dataset_val, label_val = shuffle(dataset_val, label_val, random_state=71)
            dataset_val, label_val = shuffle(dataset_val, label_val, random_state=72)
            dataset_val, label_val = shuffle(dataset_val, label_val, random_state=73)
            print("BENIGN (VAL) IMAGES : {}".format(list(label_val).count(0)))
            print("MALIGNANT (VAL) IMAGES : {}".format(list(label_val).count(1)))
            dataset_val = np.array(dataset_val)
            label_val = np.array(label_val)
            return dataset, dataset_val, label, label_val
        else :
            return dataset, label
        #X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.1, random_state=123)
        #return X_train, X_test, y_train, y_test

    def get_dataset_classification_train(self):
        datagen = ImageDataGenerator(preprocessing_function=cv_preprocessing)
        train_it = datagen.flow_from_directory(self.classification_path + "train/", classes=['benign', 'malignant'], batch_size=5,
                                               color_mode="rgb", target_size=(224, 224))
        val_it = datagen.flow_from_directory(self.classification_path + "validation/", classes=['benign', 'malignant'], batch_size=5,
                                               color_mode="rgb", target_size=(224, 224))
        return train_it, val_it

    def get_dataset_classification_test(self):
        datagen = ImageDataGenerator(preprocessing_function=cv_preprocessing)
        test_it = datagen.flow_from_directory(self.classification_path + "test/", classes=['benign', 'malignant'], batch_size=5,
                                               color_mode="rgb", target_size=(224, 224))

        return test_it