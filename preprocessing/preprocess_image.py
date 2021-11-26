from tensorflow.keras.utils import normalize
import os
import cv2
import numpy as np
from keras.models import load_model
from utils import utils

class Preprocessor():

    def __init__(self):
        self.unet = load_model("machine_learning/checkpoints/unet/unet_model_colab_3500i_30e_32bs_0005lr_relu.hdf5")

    def read_in_rgb(self,path):
        imm = cv2.imread(path)
        return cv2.cvtColor(imm, cv2.COLOR_BGR2RGB)

    def mole_detect(self, origin_image_rgb, mask):
        # if this function is call from unet_preprocessing, the contours is only one always
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour = None
        max_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            # print(area)
            if area > max_area:
                max_area = area
                contour = c
        boundRect = cv2.boundingRect(contour)
        x, y, w, h = boundRect
        f = origin_image_rgb[y:y + h, x:x + w]
        k = cv2.resize(f, (utils.IMG_SIZE_VGG, utils.IMG_SIZE_VGG), interpolation=cv2.INTER_CUBIC)
        return k/255.

    def unet_preprocessing(self, path, model):
        original = self.read_in_rgb(path)
        original = cv2.resize(original, (utils.IMG_SIZE_UNET, utils.IMG_SIZE_UNET), interpolation=cv2.INTER_CUBIC)
        img_to_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        img_norm = np.expand_dims(normalize(np.array(img_to_gray), axis=1), 2)
        img_norm = img_norm[:, :, 0][:, :, None]
        to_input = np.expand_dims(img_norm, 0)
        # Predict and threshold for values above 0.5 probability
        # Change the probability threshold to low value (e.g. 0.05) for watershed demo.
        prediction_mask = (model.predict(to_input)[0, :, :, 0] > 0.25).astype(np.uint8)
        return self.mole_detect(original, prediction_mask)

    def cv_preprocessing(self, path):
        original = self.read_in_rgb(path)

        # remove noise and hair
        #dst = cv2.fastNlMeansDenoisingColored(original, None, 10, 10, 7, 21)
        #grayScale = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        #kernel = cv2.getStructuringElement(1, (17, 17))
        #blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
        #_, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        #final_image = cv2.inpaint(original, threshold, 1, cv2.INPAINT_TELEA)

        # Convert to HSV colourspace and extract just the Saturation
        saturation = cv2.cvtColor(original, cv2.COLOR_RGB2HSV)[..., 1]
        # Find best (Otsu) threshold to divide black from white, and apply it
        ret, mask = cv2.threshold(saturation, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self.mole_detect(original, mask)