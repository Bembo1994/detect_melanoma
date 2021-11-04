from dataset.API.isic_api import ISICApi
from utils import utils
from tensorflow.keras.preprocessing import image_dataset_from_directory
from collections import defaultdict
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.keras.utils import normalize
import os
import json
import cv2
import pandas as pd
import numpy as np
import random


class Dataset():

    def __init__(self):
        self.limit = utils.LIMIT_IMAGES
        self.real_path = os.path.dirname(os.path.realpath(__file__))
        self.savePath = self.real_path+"/ISICArchive/"
        self.segmentation_path = self.savePath + "Segmentation/"
        # Initialize the API; no login is necessary for public data
        self.api = ISICApi()
        self.make_dirs(self.savePath)
        self.make_dirs(self.savePath + "benign/")
        self.make_dirs(self.savePath + "malignant/")
        self.make_dirs(self.segmentation_path)
        self.dataframe = None
        self.img_for_segmentation = set()
        self.num_sample_for_segmentation = 2000
        self.download_metadata()
        self.download_dataset()
        self.download_mask()

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

    def download_dataset(self):

        # Opening JSON file
        metadata = json.load(open(self.savePath+"metadata.json"))
        print('Checking images')
        for image in metadata:
            # controllo delle chiavi nel dizionario
            if "meta" in image and "clinical" in image["meta"] and "benign_malignant" in image["meta"]["clinical"]:
                path = ""
                # se benigno
                if image["meta"]["clinical"]["benign_malignant"] == "benign" and image["dataset"]["name"] != "SONIC":
                    path = self.savePath+"benign/"
                # se maligno
                elif image["meta"]["clinical"]["benign_malignant"] == "malignant" and image["dataset"]["name"] != "SONIC":
                    path = self.savePath+"malignant/"
                if path != "":
                    if not os.path.isfile(path+image["name"]+".jpg"):
                        imageFileOutputPath = os.path.join(path, '%s.jpg' % image['name'])
                        imageFileResp = self.api.get('image/%s/download' % image['_id'])
                        imageFileResp.raise_for_status()
                        with open(imageFileOutputPath, 'wb') as imageFileOutputStream:
                            for chunk in imageFileResp:
                                imageFileOutputStream.write(chunk)
        print("Downloaded dataset")
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
        self.dataframe = df
        self.dataframe.to_csv(self.savePath+'dataframe.csv',index=False)

    def download_mask(self):
        print("Check image for segmentation")
        for i, name in enumerate(self.dataframe["dataset.name"]):
            if name == "SONIC":
                self.img_for_segmentation.add(self.dataframe["_id"][i])
        for i, name in enumerate(self.dataframe["meta.clinical.benign_malignant"]):
            if name != "benign" and name != "malignant":
                self.img_for_segmentation.add(self.dataframe["_id"][i])
        self.make_dirs(self.segmentation_path+"mask/")
        self.make_dirs(self.segmentation_path + "img/")

        for i, img_id in enumerate(self.img_for_segmentation) :
            if len(os.listdir(self.segmentation_path+"mask/")) < self.num_sample_for_segmentation and len(os.listdir(self.segmentation_path+"mask/")) == len(os.listdir(self.segmentation_path+"img/")):
                name = self.dataframe.iloc[list(self.dataframe["_id"]).index(img_id)]["name"]
                print(name)
                flag = False
                if not os.path.isfile(self.segmentation_path+"mask/%s.tiff" % name) and not os.path.isfile(self.segmentation_path+"img/%s.tiff" % name):
                    flag = True

                if flag :
                    segmentation = self.api.getJson('segmentation?imageId=' + img_id)
                    if len(segmentation)>0:
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

    def get_train_val_set(self):
        ds_train = image_dataset_from_directory(
            self.savePath,
            labels='inferred',
            label_mode="binary",
            image_size=[utils.IMG_SIZE, utils.IMG_SIZE],
            interpolation='nearest',
            batch_size=16,
            shuffle=True,
            validation_split=0.3,
            subset='training',
            seed=123
        )

        ds_val = image_dataset_from_directory(
            self.savePath,
            labels='inferred',
            label_mode="binary",
            image_size=[utils.IMG_SIZE, utils.IMG_SIZE],
            interpolation='nearest',
            batch_size=16,
            shuffle=True,
            validation_split=0.3,
            subset='validation',
            seed=123
        )

        return (ds_train,ds_val)

    def get_train_test_set_for_segmentation(self):
        print("Load dataset for UNet")
        image_directory = self.segmentation_path+'img/'
        mask_directory = self.segmentation_path+'mask/'

        size = utils.IMG_SIZE_UNET
        image_dataset = []  # Many ways to handle data, you can use pandas. Here, we are using a list format.
        mask_dataset = []  # Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

        images = os.listdir(image_directory)
        for i, image_name in enumerate(images):  # Remember enumerate method adds a counter and returns the enumerate object
            if (image_name.split('.')[1] == 'tiff'):
                # print(image_directory+image_name)
                image = cv2.imread(image_directory + image_name, 0)
                image = Image.fromarray(image)
                image = image.resize((size, size))
                image_dataset.append(np.array(image))

        masks = os.listdir(mask_directory)
        for i, image_name in enumerate(masks):
            if (image_name.split('.')[1] == 'tiff'):
                image = cv2.imread(mask_directory + image_name, 0)
                image = Image.fromarray(image)
                image = image.resize((size, size))
                mask_dataset.append(np.array(image))
        print("Dataset loaded")
        # Normalize images
        image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1), 3)
        # D not normalize masks, just rescale to 0 to 1.
        mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) / 255.

        X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.10, random_state=0)

        return X_train, X_test, y_train, y_test
        #return ((image_dataset.shape[1], image_dataset.shape[2], image_dataset.shape[3]),(X_train, X_test, y_train, y_test))
