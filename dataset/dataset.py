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
        return self.get_dataset(self.segmentation_path + 'Dataset_Test_Segmentation_Dict_' + str(utils.TEST_SET_SIZE) + '.pkl')

    def get_train_and_val_set_for_segmentation(self):
        return self.get_dataset(self.segmentation_path + 'Dataset_Segmentation_Dict_Colab_' + str(utils.LIMIT_IMAGES_SEGMENTATION_PKL) + '.pkl')





    '''
    
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
    
    '''

    '''
    def remove_test_img(self, images_shuffled, mask_shuffled):
        l1 = set(images_shuffled)
        l2 = set(mask_shuffled)
        l3 = set(self.name_imgs_training_set_unet["img"])
        l4 = set(self.name_imgs_training_set_unet["mask"])
        if l3 == l4 :
            return list(l1-l3), list(l2-l4)
        else :
            sys.exit('Error in splitting')
    '''

    '''

    def get_train_val_set_for_segmentation(self, iterator):
    start = utils.IMG_IN_RAM * iterator
    end = utils.IMG_IN_RAM * (iterator + 1)

    print("Load dataset for UNet")
    image_directory = self.segmentation_path+'img/'
    mask_directory = self.segmentation_path+'mask/'

    size = utils.IMG_SIZE_UNET
    image_dataset = []  # Many ways to handle data, you can use pandas. Here, we are using a list format.
    mask_dataset = []  # Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

    images = os.listdir(image_directory)
    masks = os.listdir(mask_directory)

    #shuffling
    #shuffle_tmp = list(zip(images, masks))
    #random.shuffle(shuffle_tmp)
    #images_shuffled, mask_shuffled = zip(*shuffle_tmp)
    #images_shuffled = images
    #mask_shuffled = masks

    if len(self.name_imgs_training_set_unet) > 0 :
        images_shuffled, mask_shuffled = self.remove_test_img(images_shuffled, mask_shuffled)
    else :
        self.set_name_img_training_set_unet(images_shuffled, mask_shuffled)
        images_shuffled, mask_shuffled = self.remove_test_img(images_shuffled, mask_shuffled)

    #print(images)
    #print("OOOO")
    #print(images[start:end])
    if len(self.name_imgs_training_set_unet) == 0:
        self.set_name_img_training_set_unet(images, masks)
    #print(images_shuffled[start:end])
    for i, image_name in enumerate(images[start:end]):
    #for i, image_name in enumerate(images_shuffled):
        if (image_name.split('.')[1] == 'tiff' and not image_name in self.name_imgs_training_set_unet["mask"] and not image_name in self.name_imgs_training_set_unet["img"]):
            #print(image_directory + image_name)
            image = cv2.imread(image_directory + image_name, 0)
            image = Image.fromarray(image)
            image = image.resize((size, size))
            image_dataset.append(np.array(image))

    for i, image_name in enumerate(masks[start:end]):
        if (image_name.split('.')[1] == 'tiff' and not image_name in self.name_imgs_training_set_unet["mask"] and not image_name in self.name_imgs_training_set_unet["img"]):
            image = cv2.imread(mask_directory + image_name, 0)
            image = Image.fromarray(image)
            image = image.resize((size, size))
            mask_dataset.append(np.array(image))
    print("Dataset loaded")
    # Normalize images
    image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1), 3)
    # D not normalize masks, just rescale to 0 to 1.
    mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) / 255.

    X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.1, random_state=0)

    dataset_dict = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
    print("writing")
    with open(self.segmentation_path+'dataset_dict.pkl', 'wb') as file:
        pickle.dump(dataset_dict, file)
    print("write")
    return X_train, X_test, y_train, y_test
    #return ((image_dataset.shape[1], image_dataset.shape[2], image_dataset.shape[3]),(X_train, X_test, y_train, y_test))
    
    '''

    '''
    def set_name_img_training_set_unet(self,images, masks):

    if len(images) != len(masks):
        sys.exit('Error in dataset')
    N = len(images)

    len_test_set = N%utils.IMG_IN_RAM + utils.IMG_IN_RAM

    for i in range(len_test_set):
        random_img = random.choice(images)
        while random_img in self.name_imgs_training_set_unet["mask"] and random_img in self.name_imgs_training_set_unet["img"]:
            random_img = random.choice(images)
        self.name_imgs_training_set_unet["mask"].append(random_img)
        self.name_imgs_training_set_unet["img"].append(random_img)

    '''