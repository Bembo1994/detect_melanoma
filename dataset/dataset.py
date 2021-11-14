from dataset.API.isic_api import ISICApi
from utils import utils
from tensorflow.keras.preprocessing import image_dataset_from_directory
from collections import defaultdict
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.keras.utils import normalize
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
        self.name_imgs_training_set_unet = defaultdict(list)
        self.img_for_segmentation = set()
        self.num_sample_for_segmentation = utils.LIMIT_IMAGES_SEGMENTATION_DOWNLOAD
        self.image_directory = self.segmentation_path + 'img/'
        self.mask_directory = self.segmentation_path + 'mask/'

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
        metadata = json.load(open(self.savePath+"metadata.json"))
        print('Checking images')
        if ((len(os.listdir(self.savePath+"benign/")) + len(os.listdir(self.savePath+"malignant/"))) < self.limit) :
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
            df.to_csv(self.savePath+'dataframe_cleaned.csv',index=False)

    def download_mask(self):
        if not os.path.isfile(self.savePath+'dataframe.csv'):
            sys.exit("ERROR : Before download the dataset")
        df = pd.read_csv(self.savePath+'dataframe.csv')
        print("Check {} images for segmentation".format(self.num_sample_for_segmentation))

        #scorro il dataframe e trovo 5K immagini (limit images segmentation download) da scaricare dalle ~16K con campo bening or malignant nullo
        for i, name in enumerate(df["dataset.name"]):
            if name == "SONIC":
                self.img_for_segmentation.add(df["_id"][i])

        for i, name in enumerate(df["meta.clinical.benign_malignant"]):
            if name != "benign" and name != "malignant":
                self.img_for_segmentation.add(df["_id"][i])

        self.make_dirs(self.segmentation_path+"mask/")
        self.make_dirs(self.segmentation_path + "img/")

        #se non sono state scaricate tutte le immagini allora continua a scaricarle
        if len(os.listdir(self.segmentation_path + "mask/")) < self.num_sample_for_segmentation and len(os.listdir(self.segmentation_path + "mask/")) == len(os.listdir(self.segmentation_path + "img/")):

            for i, img_id in enumerate(self.img_for_segmentation) :
                name = df.iloc[list(df["_id"]).index(img_id)]["name"]
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

    # if flag is true then is for train and validation else is for test
    def dataset_segmentation_to_pickle(self, start, end, flag):

        if flag:
            test_size = 1
            name_file = 'Dataset_Test_Segmentation_Dict_' + str(end) + '.pkl'
        else:
            test_size = 0.1
            name_file = 'Dataset_Segmentation_Dict_Colab_' + str(end) + '.pkl'

        if os.path.isfile(self.segmentation_path + name_file):
            print("File pickle is already written")
            return

        print("Creating pickle dataset for UNet")

        size = utils.IMG_SIZE_UNET
        images = os.listdir(self.image_directory)
        masks = os.listdir(self.mask_directory)

        image_dataset = []
        mask_dataset = []
        for image_name in images[start:start + end]:
            image = cv2.imread(self.image_directory + image_name, 0)
            image = Image.fromarray(image)
            image = image.resize((size, size))
            image_dataset.append(np.array(image))

        for image_name in masks[start:start + end]:
            image = cv2.imread(self.mask_directory + image_name, 0)
            image = Image.fromarray(image)
            image = image.resize((size, size))
            mask_dataset.append(np.array(image))

        print("Dataset loaded")
        # Normalize images
        image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1), 3)
        # D not normalize masks, just rescale to 0 to 1.
        mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) / 255.

        X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=test_size,
                                                            random_state=0)

        print("Writing pickle")

        dataset_dict = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
        with open(self.segmentation_path + name_file, 'wb') as f:
            pickle.dump(dataset_dict, f)

        print("Write")

    def get_dataset(self, path):
        with open(path, "rb") as f:
            tmp = pickle.load(f)
        X_train = tmp["X_train"]
        X_test = tmp["X_test"]
        y_train = tmp["y_train"]
        y_test = tmp["y_test"]
        return X_train, X_test, y_train, y_test

    def get_test_set_for_segmentation(self):
        return self.get_dataset(self.segmentation_path + 'Dataset_Test_Segmentation_Dict_' + str(utils.TEST_SET_SIZE_SEGMENTATION) + '.pkl')

    def get_train_and_val_set_for_segmentation(self):
        return self.get_dataset(self.segmentation_path + 'Dataset_Segmentation_Dict_Colab_' + str(utils.LIMIT_IMAGES_SEGMENTATION_PKL) + '.pkl')

    #flag is for cv_preprocessing
    def classification_pickle(self, flag, names_benign, names_malignant, unet_model, preprocessor, test_size):

        for count in range(4) :

            start = count * 1000
            end = (count + 1) * 1000

            if flag and test_size<1:
                name_file = "dataset_dict_classification_(CV)_{}i_10%.pkl".format(end)
            if flag and test_size==1:
                name_file = "dataset_test_dict_classification_(CV)_{}i_10%.pkl".format(end)
            if not flag and test_size < 1:
                name_file = "dataset_dict_classification_(UNET_{}i_{}e_{}bs_{}lr_{})_{}i_10%.pkl".format(utils.LIMIT_IMAGES_SEGMENTATION_PKL, utils.EPOCHS_UNET, utils.BS_UNET, str(utils.LR_UNET).split(".")[1], utils.FUNCTION_UNET, end)
            if not flag and test_size == 1:
                name_file = "dataset_test_dict_classification_(UNET_{}i_{}e_{}bs_{}lr_{})_{}i_10%.pkl".format(utils.LIMIT_IMAGES_SEGMENTATION_PKL, utils.EPOCHS_UNET, utils.BS_UNET, str(utils.LR_UNET).split(".")[1], utils.FUNCTION_UNET, end)

            if not os.path.isfile(self.savePath + name_file):
                dataset = []
                label = []

                j = 0

                for image_name in names_benign[start:end]:
                    if j % 1000 == 0:
                        print(j)
                    path_img = self.savePath + "benign/" + image_name + ".jpg"
                    if os.path.isfile(path_img):
                        try: # try segmentation else get the total image and resize
                            if flag:
                                _, _, result = preprocessor.cv_preprocessing(path_img, 0)
                            else :
                                _, _, result = preprocessor.unet_preprocessing(path_img, unet_model, 0)
                            dataset.append(np.array(result))
                            label.append(0)
                            j += 1
                        except:
                            img = preprocessor.read_in_rgb(path_img)
                            img = cv2.resize(img, (utils.IMG_SIZE_VGG, utils.IMG_SIZE_VGG), interpolation=cv2.INTER_CUBIC)
                            dataset.append(np.array(img))
                            label.append(0)
                            j += 1
                            if flag:
                                print("Error in cv_preprocessing with image (malignant) : {}".format(image_name))
                            else :
                                print("Error in unet_preprocessing with image (malignant) : {}".format(image_name))
                j = 0
                for image_name in names_malignant[start:end]:
                    if j % 1000 == 0:
                        print(j)
                    path_img = self.savePath + 'malignant/' + image_name + ".jpg"
                    if os.path.isfile(path_img):
                        try: # try segmentation else get the total image and resize
                            if flag:
                                _, _, result = preprocessor.cv_preprocessing(path_img, 0)
                            else :
                                _, _, result = preprocessor.unet_preprocessing(path_img, unet_model, 0)
                            dataset.append(np.array(cv_result))
                            label.append(1)
                            j += 1
                        except:
                            img = preprocessor.read_in_rgb(path_img)
                            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                            dataset.append(np.array(img))
                            label.append(1)
                            j += 1
                            if flag:
                                print("Error in cv_preprocessing with image (malignant) : {}".format(image_name))
                            else :
                                print("Error in unet_preprocessing with image (malignant) : {}".format(image_name))

                dataset = np.array(dataset)
                label = np.array(label)

                X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=test_size, random_state=0)
                print("Writing pickle")

                dataset_dict = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

                with open(self.savePath + name_file, 'wb') as f:
                    pickle.dump(dataset_dict, f)

                print("Write")

    def dataset_classification_to_pickle(self, flag):
        preprocessor = preprocess_image.Preprocessor()
        unet = neural_network.UNet()
        unet_model = unet.get_model(utils.IMG_SIZE_UNET, utils.IMG_SIZE_UNET, utils.IMG_CHANNELS_UNET)
        df = pd.read_csv(self.savePath+'dataframe_cleaned.csv')

        names_benign = []
        names_malignant = []
        for i, b_or_m in enumerate(df["meta.clinical.benign_malignant"]):
            if b_or_m == "benign":
                names_benign.append(df.iloc[i]["name"])
            elif b_or_m == "malignant":
                names_malignant.append(df.iloc[i]["name"])

        if not flag: #if false then make the training and validation set

            self.classification_pickle(flag=True, names_benign=names_benign, names_malignant=names_malignant, unet_model=None, preprocessor=preprocessor, test_size=0.10)
            self.classification_pickle(flag=False, names_benign=names_benign, names_malignant=names_malignant, unet_model=unet_model, preprocessor=preprocessor, test_size=0.10)

        else : #else test set

            self.classification_pickle(flag=True, names_benign=names_benign, names_malignant=names_malignant, unet_model=None, preprocessor=preprocessor, test_size=1)
            self.classification_pickle(flag=False, names_benign=names_benign, names_malignant=names_malignant, unet_model=unet_model, preprocessor=preprocessor, test_size=1)

    #flag is for cv
    def get_train_and_val_set_for_classification(self, flag_unet, flag_test):
        X_train1, X_test1, y_train1, y_test1 = self.get_dataset( self.savePath + "/Classification/dict_train_val_classification_CV_0-1500i_20%.pkl")
        X_train2, X_test2, y_train2, y_test2 = self.get_dataset( self.savePath + "/Classification/dict_train_val_classification_CV_1500-3000i_20%.pkl")
        X_train = np.concatenate((X_train1, X_train2))
        X_test = np.concatenate((X_test1, X_test2))
        y_train = np.concatenate((y_train1, y_train2))
        y_test = np.concatenate((y_test1, y_test2))
        return X_train, X_test, y_train, y_test
        '''
        if flag_unet and flag_test:
            return self.get_dataset(self.savePath + "dataset_test_dict_segmentation_(CV)_{}i_10%.pkl".format(utils.LIMIT_IMAGES_CLASSIFICATION_PKL))
        if flag_unet and not flag_test:
            return self.get_dataset(self.savePath + "dataset_dict_segmentation_(CV)_{}i_10%.pkl".format(utils.LIMIT_IMAGES_CLASSIFICATION_PKL))
        if not flag_unet and flag_test:
            return self.get_dataset(self.savePath + "dataset_test_dict_segmentation_(UNET_{}i_{}e_{}bs_{}lr_{})_{}i_10%.pkl".format(utils.LIMIT_IMAGES_SEGMENTATION_PKL, utils.EPOCHS_UNET, utils.BS_UNET, str(utils.LR_UNET).split(".")[1], utils.FUNCTION_UNET, utils.LIMIT_IMAGES_CLASSIFICATION_PKL))
        if not flag_unet and not flag_test:
            return self.get_dataset(self.savePath + "dataset_dict_segmentation_(UNET_{}i_{}e_{}bs_{}lr_{})_{}i_10%.pkl".format(utils.LIMIT_IMAGES_SEGMENTATION_PKL, utils.EPOCHS_UNET, utils.BS_UNET, str(utils.LR_UNET).split(".")[1], utils.FUNCTION_UNET, utils.LIMIT_IMAGES_CLASSIFICATION_PKL))
        '''





