from dataset.API.isic_api import ISICApi
from utils import utils
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_image
from PIL import Image
from tensorflow.keras.utils import normalize
import glob
import os
import json
import cv2
import pandas as pd
import numpy as np
import random
import sys
import pickle
from sklearn.utils import shuffle
import itertools

class Dataset():

    def __init__(self, cv_or_unet):
        self.real_path = os.path.dirname(os.path.realpath(__file__))
        self.savePath = self.real_path+"/ISICArchive/"
        self.segmentation_path = self.savePath + "Segmentation/"
        self.mask_directory = self.segmentation_path + "mask/"
        self.image_directory = self.segmentation_path + "img/"
        self.classification_path = self.savePath + "Classification/"
        self.xai_path = self.savePath + "Explainable_ai_abcd/"
        self.api = ISICApi()
        self.make_dirs(self.savePath)
        self.make_dirs(self.segmentation_path)
        self.make_dirs(self.classification_path)
        self.preprocessor = preprocess_image.Preprocessor()
        self.size_test_set = utils.SIZE_TEST_SET
        self.cv_or_unet = cv_or_unet
        self.limit = utils.LIMIT_IMAGES
        self.limit_seg = utils.LIMIT_IMAGES_SEGMENTATION_DOWNLOAD
        self.download_metadata()
        self.download_dataset_classification()
        self.download_dataset_segmentation()

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
            print("Saved file : {}".format(self.savePath + "dataframe.csv"))
        else:
            print("File already saved : {}".format(self.savePath + "dataframe.csv"))

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
            df_c = df[df['meta.acquisition.image_type'].notna()].set_index('meta.acquisition.image_type').drop(["clinical"]).reset_index().set_index("dataset.name").drop("SONIC").reset_index()
            df_c.to_csv(self.savePath + 'dataframe_cleaned_classification.csv', index=True)
            print("Saved file : {}".format(self.savePath + "dataframe_cleaned_classification.csv"))

            ds_name = list(df["dataset.name"].unique())
            drop_names = list(set(ds_name) - set(["SONIC"]))
            df_s = df.set_index("dataset.name")
            df_s = df_s.drop(drop_names)
            df_s = df_s.reset_index()
            df_s.set_index("_id")
            df_s.to_csv(self.savePath + 'dataframe_cleaned_segmentation.csv', index=True)
            print("Saved file : {}".format(self.savePath + "dataframe_cleaned_segmentation.csv"))
        else:
            print("File already saved : {}".format(self.savePath + "dataframe_cleaned_classification.csv"))
            print("File already saved : {}".format(self.savePath + "dataframe_cleaned_segmentation.csv"))

    def download_dataset_classification(self):

        if os.path.isfile(self.savePath+'dataframe_cleaned_classification.csv'):
            df = pd.read_csv(self.savePath+'dataframe_cleaned_classification.csv')
        else:
            sys.exit("ERROR : Dataframe is not saved")
        if len(os.listdir(self.classification_path)) > 0:
            print("Dataset already download")
            return

        #df = df.set_index("dataset.name")
        #df = df.drop(["SONIC"])     #the images of this class are full of artifacts, good for segmentation
        #df = df.reset_index()

        malignants = df.set_index("meta.clinical.benign_malignant").drop("benign").set_index("_id")
        benigns = df.set_index("meta.clinical.benign_malignant").drop("malignant").set_index("_id").sample(len(malignants))

        test_malignants = malignants.sample(175)
        test_benings = benigns.sample(175)

        train_malignants = malignants.drop(set(test_malignants.reset_index()["_id"]))
        train_benigns = benigns.drop(set(test_benings.reset_index()["_id"]))

        test_malignants = test_malignants.reset_index()
        test_benings = test_benings.reset_index()
        train_malignants = train_malignants.reset_index()
        train_benigns = train_benigns.reset_index()

        def download(df, path):
            print('Checking images')
            for i, name in enumerate(df["name"]):
                if i%1000 == 0 and i > 0:
                    print("Images download {}".format(i))
                name_ext = str(name)+".jpg"
                id = df.iloc[i]["_id"]
                #b_or_m = df.iloc[i]["meta.clinical.benign_malignant"]
                #path = self.classification_path + b_or_m + "/"
                imageFileOutputPath = os.path.join(path, name_ext)
                if not (os.path.isfile((imageFileOutputPath))):
                    imageFileResp = self.api.get('image/%s/download' % id)
                    imageFileResp.raise_for_status()
                    with open(imageFileOutputPath, 'wb') as imageFileOutputStream:
                        for chunk in imageFileResp:
                            imageFileOutputStream.write(chunk)
            print("Downloaded dataset")

        self.make_dirs(self.classification_path+"train/benign/")
        self.make_dirs(self.classification_path + "train/malignant/")
        self.make_dirs(self.classification_path+"test/benign/")
        self.make_dirs(self.classification_path + "test/malignant/")

        download(test_malignants, self.classification_path + "test/malignant/")
        download(test_benings, self.classification_path + "test/benign/")
        download(train_malignants, self.classification_path + "train/malignant/")
        download(train_benigns, self.classification_path + "train/benign/")

        print("Finish download")

    def download_dataset_segmentation(self):
        if not os.path.isfile(self.savePath+'dataframe_cleaned_segmentation.csv') :
            sys.exit("ERROR : Before download the dataset")
        if len(os.listdir(self.mask_directory)) == len(os.listdir(self.image_directory)): #self.image_directory eliminata
            if len(os.listdir(self.mask_directory)) > 0 and len(os.listdir(self.image_directory)) > 0:
                print("Dataset for segmentation already download")
                return

        df = pd.read_csv(self.savePath+'dataframe_cleaned_segmentation.csv').sample(self.limit_seg)
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

                    imageFileOutputPath_seg = os.path.join(self.mask_directory, '%s.tiff' % name)
                    imageFileOutputPath_img = os.path.join(self.image_directory, '%s.tiff' % name)

                    with open(imageFileOutputPath_seg, 'wb') as imageFileOutputStream_seg:
                        for chunk in imageFileResp_seg:
                            imageFileOutputStream_seg.write(chunk)
                    with open(imageFileOutputPath_img, 'wb') as imageFileOutputStream_img:
                        for chunk in imageFileResp_img:
                            imageFileOutputStream_img.write(chunk)

        print("Downloaded image for segmentation")

    def data_augmentation(self):
        def aug(dir, path):
            name_class = path.split("/")[-2]
            for i in dir:
                img = self.preprocessor.read_in_rgb(self.classification_path + "train/"+name_class+"/" + i)
                degrees = [90, 180, 270]
                darkness = [20, 35, 50, 60]
                flippings = [1, 0, -1]
                combinations = list(itertools.product(degrees, darkness, flippings))
                random.shuffle(combinations)
                for _ in range(3):
                    c = random.choice(combinations)
                    (h, w) = img.shape[:2]
                    (cX, cY) = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D((cX, cY), c[0], 1.0)
                    rotated = cv2.warpAffine(img, M, (w, h))
                    adjusted = cv2.convertScaleAbs(rotated, alpha=1, beta=c[1])
                    flipped = cv2.flip(adjusted, c[2])
                    combinations.remove(c)
                    name = "train/"+name_class+"/rotation_{}_dark_{}_flip_{}_".format(c[0], c[1], c[2])
                    cv2.imwrite(self.classification_path + name + i, cv2.cvtColor(flipped, cv2.COLOR_RGB2BGR))
        data_to_aug = os.listdir(self.classification_path+"train/benign/")
        aug(data_to_aug, self.classification_path+"train/malignant/")
        data_to_aug = os.listdir(self.classification_path+"train/benign/")
        aug(data_to_aug, self.classification_path+"train/benign/")

    def get_dataset_segmentation(self, is_test):
        if is_test:
            image_directory = self.segmentation_path+'test/img/'
            mask_directory = self.segmentation_path+'test/mask/'
        else:
            image_directory = self.segmentation_path+'train/img/'
            mask_directory = self.segmentation_path+'train/mask/'

        image_names = glob.glob(image_directory+"*.tiff")
        mask_names = glob.glob(mask_directory + "*.tiff")
        image_dataset = []
        print(len(image_names))
        for k in image_names:
            image = cv2.imread(k, 0)
            image = Image.fromarray(image)
            image = image.resize((256, 256))
            image_dataset.append(np.array(image))

        mask_dataset = []
        for k in mask_names:
            image = cv2.imread(k, 0)
            image = Image.fromarray(image)
            image = image.resize((256, 256))
            mask_dataset.append(np.array(image))

        # Normalize images
        image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1), 3)
        # D not normalize masks, just rescale to 0 to 1.
        mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) / 255.
        if is_test:
            return image_dataset, mask_dataset
        X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.20,
                                                            random_state=42)

        return X_train, X_test, y_train, y_test

    def get_dataset_classification(self, network, is_test, xai_abcd_flag):
        self.preprocessor.net = network

        if network == "vgg16":
            size = (utils.IMG_SIZE_VGG,utils.IMG_SIZE_VGG)
        elif network == "resnet":
            size = (utils.IMG_SIZE_RESNET,utils.IMG_SIZE_RESNET)
        elif network == "inception_v3":
            size = (utils.IMG_SIZE_INCEPTION,utils.IMG_SIZE_INCEPTION)
        else:
            sys.exit("Error in size images")

        self.preprocessor.size = size
        image_dataset = []
        label_dataset = []
        names = []

        if is_test:
            app = "test/"
        else:
            app = "train/"
            if os.path.isfile(self.classification_path+'train_{}.pickle'.format(self.cv_or_unet, size[0])):
                with open(self.classification_path+'train_{}.pickle'.format(self.cv_or_unet, size[0]), 'rb') as handler:
                    d = pickle.load(handler)
                    print("Dataset già esistente, caricamento")
                    X_train, X_test, y_train, y_test = d["X_train"], d["X_test"], d["y_train"], d["y_test"]
                for i, image in enumerate(X_train):
                    X_train[i] = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
                for i, image in enumerate(X_test):
                    X_test[i] = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
                return X_train, X_test, y_train, y_test
        if xai_abcd_flag:
            path_benign_directory = self.xai_path + 'benign/'
            path_malignant_directory = self.xai_path + 'malignant/'
        else:
            path_benign_directory = self.classification_path + app + 'benign/'
            path_malignant_directory = self.classification_path + app + 'malignant/'
        benigns = os.listdir(path_benign_directory)
        malignants = os.listdir(path_malignant_directory)
        for i, b in enumerate(benigns):
            if i%1000 == 0:
                print(i)
            if self.cv_or_unet == "threshold":
                k = self.preprocessor.cv_preprocessing(path_benign_directory+b)
            else:
                k = self.preprocessor.unet_preprocessing(path_benign_directory+b)
            image_dataset.append(k)
            label_dataset.append(0)
            names.append(b)

        for i, m in enumerate(malignants):
            if i%1000==0:
                print(i)
            if self.cv_or_unet == "threshold":
                k = self.preprocessor.cv_preprocessing(path_malignant_directory+m)
            else:
                k = self.preprocessor.unet_preprocessing(path_malignant_directory+m)
            image_dataset.append(k)
            label_dataset.append(1)
            names.append(m)
        tmp = list(zip(image_dataset, label_dataset, names))
        random.shuffle(tmp)
        image_dataset, label_dataset, names = zip(*tmp)
        image_dataset = np.array(image_dataset)
        label_dataset = np.array(label_dataset)
        if is_test:
            return image_dataset, label_dataset, names
        X_train, X_test, y_train, y_test = train_test_split(image_dataset, label_dataset, test_size=0.20,
		                                        random_state=42)
        ds_to_pickle = {"X_train":X_train, "X_test":X_test, "y_train":y_train, "y_test":y_test}
        with open(self.classification_path+'train_{}_{}.pickle'.format(self.cv_or_unet, size[0]), 'wb') as handle:
            pickle.dump(ds_to_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return X_train, X_test, y_train, y_test

    def get_all_img(self, start_point):
        self.preprocessor.net = network
        if network == "vgg16":
            size = (utils.IMG_SIZE_VGG,utils.IMG_SIZE_VGG)
        elif network == "resnet":
            size = (utils.IMG_SIZE_RESNET,utils.IMG_SIZE_RESNET)
        elif network == "inception_v3":
            size = (utils.IMG_SIZE_INCEPTION,utils.IMG_SIZE_INCEPTION)
        else:
            sys.exit("Error in size images")
        self.preprocessor.size = size
        image_dataset = []
        label_dataset = []
        names = []
        path_benign_directory = '/home/bembo/Scrivania/detect_melanoma/dataset/ISICArchive/Classification_totali/train/benign/'
        path_malignant_directory = '/home/bembo/Scrivania/detect_melanoma/dataset/ISICArchive/Classification_totali/train/malignant/'
        benigns = os.listdir(path_benign_directory)[start_point:start_point + 100]
        malignants = os.listdir(path_malignant_directory)[start_point:start_point + 100]
        for i, b in enumerate(benigns):
            if i%50 == 0:
                print(i)
            if self.cv_or_unet == "threshold":
                k = self.preprocessor.cv_preprocessing(path_benign_directory+b)
            else:
                k = self.preprocessor.unet_preprocessing(path_benign_directory+b)
            image_dataset.append(k)
            label_dataset.append(0)
            names.append(b)

        for i, m in enumerate(malignants):
            if i%50==0:
                print(i)
            if self.cv_or_unet == "threshold":
                k = self.preprocessor.cv_preprocessing(path_malignant_directory+m)
            else:
                k = self.preprocessor.unet_preprocessing(path_malignant_directory+m)
            image_dataset.append(k)
            label_dataset.append(1)
            names.append(m)
        tmp = list(zip(image_dataset, label_dataset, names))
        random.shuffle(tmp)
        image_dataset, label_dataset, names = zip(*tmp)
        image_dataset = np.array(image_dataset)
        label_dataset = np.array(label_dataset)
        return image_dataset, label_dataset, names