import urllib
import os
import csv
import json
import shutil
import cv2
import pandas as pd
import numpy as np
import torchvision
from dataset.API.isic_api import ISICApi #from isic_api import ISICApi
from utils import utils
from torchvision import transforms
from torch.utils.data import DataLoader

class Dataset():

    def __init__(self):
        self.limit = utils.LIMIT_IMAGES
        self.real_path = os.path.dirname(os.path.realpath(__file__))
        self.savePath = self.real_path+"/ISICArchive/"
        # Initialize the API; no login is necessary for public data
        self.api = ISICApi()
        self.make_dirs(self.savePath)
        self.make_dirs(self.savePath + "benign/")
        self.make_dirs(self.savePath + "malignant/")
        self.download_metadata()
        self.download_dataset()

    def make_dirs(self,path):
        if not os.path.exists(path):
            print("Make directories : {}".format(path))
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
        print('Downloading images')
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

    def dataframe_analysis(self):
        print("Normalize metadata")
        with open(self.savePath+"metadata.json") as f:
            df = pd.read_json(f)
        column_to_expand = ["creator", "dataset", "meta", "notes"]
        for column in column_to_expand:
            data_normalize = pd.json_normalize(df[column])
            for c in data_normalize.columns:
                data_normalize.rename(columns={c: column + "." + c}, inplace=True)
            df.drop(column, axis=1, inplace=True)
            df = pd.concat([df, data_normalize], axis=1, join="inner")

        # Analizziamo l'età
        ages = df["meta.clinical.age_approx"]
        benign_malignant = df["meta.clinical.benign_malignant"]
        bening = []
        malignant = []

        print("Head of dataframe : \n{}\n\n".format(df.head()))
        print("All the keys :\n{}\n\n".format(df.keys()))
        print("Keys with null value :\n{}\n\n".format(df.isnull().sum()))
        print("Unique value in ages :\n{}\n\n".format(ages.unique()))
        print("Unique value in benign_malignant column :\n{}\n\n".format(benign_malignant.unique()))

        for i, a in enumerate(ages):
            b_or_m = benign_malignant[i]
            if str(a) != "nan" and str(b_or_m) != "None" and str(b_or_m) != "nan" and str(
                    b_or_m) != "indeterminate/malignant" and str(b_or_m) != "indeterminate" and str(
                    b_or_m) != "indeterminate/benign":
                if b_or_m == "benign":
                    bening.append(a)
                else:
                    malignant.append(a)

        print("Mean age for mole bening :\n{}\n\n".format(sum(bening) / len(bening)))
        print("Mean age for mole malignant :\n{}\n\n".format(sum(malignant) / len(malignant)))

        df["mean"] = np.nan
        df["std"] = np.nan

        # Calcoliamo media, mediana e dev stand delle immagini e le aggiungiamo al df
        names = list(df["name"])
        dim = (utils.IMG_SIZE, utils.IMG_SIZE)
        if not os.path.isfile(self.savePath+"dataframe.pkl"):
            df_copy = df.copy()
            dirs = ["benign","malignant"]
            for dir in dirs:
                print(dir)
                for img_name in os.listdir(self.savePath + dir):
                    if img_name[:-4] in names:
                        pos = names.index(img_name[:-4])
                        img = cv2.imread(self.savePath + dir + "/" + img_name)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
                        mean, std = cv2.meanStdDev(img)
                        df_copy.iloc[pos, df_copy.columns.get_loc('mean')] = mean[0][0]
                        df_copy.iloc[pos, df_copy.columns.get_loc('std')] = std[0][0]
            df_copy.to_pickle(self.savePath+"dataframe.pkl")
            print(df_copy)
            print("Dataframe saved")
            return df_copy
        else:
            return pd.read_pickle(self.savePath + "dataframe.pkl")

    def dataset_analysis(self):

        transform_img = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])

        image_data = torchvision.datasets.ImageFolder(
            root=self.savePath, transform=transform_img
        )

        image_data_loader = DataLoader(
            image_data,
            batch_size=len(image_data),
            shuffle=False,
            num_workers=0
        )

        #images, labels = next(iter(image_data_loader))

        def mean_std(loader):
            images, lebels = next(iter(loader))
            # shape of images = [b,c,w,h]
            mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])
            return mean, std

        mean, std = mean_std(image_data_loader)
        print("mean and std: \n", mean, std)