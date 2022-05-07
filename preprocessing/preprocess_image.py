from tensorflow.keras.utils import normalize
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils import utils

class Preprocessor():

    def __init__(self):
        self.delta = utils.DELTA_PREPROCESSING
        self.net = "inception_v3"
        self.size = (299, 299)

    def read_in_rgb(self, path):
        imm = cv2.imread(path)
        return cv2.cvtColor(imm, cv2.COLOR_BGR2RGB)

    def mole_detect(self, origin_image_rgb, mask, delta):

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour = None
        max_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > max_area:
                max_area = area
                contour = c
        boundRect = cv2.boundingRect(contour)
    
        output = cv2.bitwise_and(origin_image_rgb, origin_image_rgb, mask=mask)

        r = cv2.drawContours(image=output, contours=contour, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        x1, y1, w, h = boundRect
    
        x2 = x1 + w + delta
        y2 = y1 + h + delta
    
        if x1 - delta <= 0 : 
            x1 = 0
        else :
            x1 = x1 - delta

        if x2 >= origin_image_rgb.shape[0]:
            x2 = origin_image_rgb.shape[0]# - w - delta
    
        if y1 - delta <= 0 : 
            y1 = 0
        else :
            y1 = y1 - delta

        if y2 >= origin_image_rgb.shape[1]:
            y2 = origin_image_rgb.shape[1]# - h - delta
    
        bby = y2 - y1
        bbx = x2 - x1
    
        if bbx/bby < 1:
            diff = bby - bbx
            while diff > 0 :
                count_tmp = 0
                if x1 - 1 > 0 and diff - (count_tmp + 1) >= 0:
                    x1 -= 1
                    count_tmp += 1
                if x2 + 1 <= origin_image_rgb.shape[0] and diff - (count_tmp + 1) >= 0:
                    x2 += 1
                    count_tmp += 1
                
                if count_tmp > 1:
                    diff -= count_tmp
                else:
                    diff -= 1          
        bby = y2 - y1
        bbx = x2 - x1
    
        if bbx/bby > 1:
            diff = bbx - bby
            while diff > 0 :
                count_tmp = 0
                if y1 - 1 > 0 and diff - (count_tmp + 1) >= 0:
                    y1 -= 1
                    count_tmp += 1
                if y2 + 1 <= origin_image_rgb.shape[1] and diff - (count_tmp + 1) >= 0:
                    y2 += 1
                    count_tmp += 1
                
                if count_tmp > 1:
                    diff -= count_tmp
                else:
                    diff -= 1

        bby = y2 - y1
        bbx = x2 - x1
                    
        rectangle = cv2.rectangle(r, (x1, y1), (x2, y2), (0, 255, 0), 2)
        f = origin_image_rgb[y1:y2, x1:x2]
        k = cv2.resize(f, self.size, interpolation=cv2.INTER_CUBIC)
        norm_image = cv2.normalize(k, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return norm_image

    def unet_preprocessing(self, path):
        unet = load_model("machine_learning/checkpoints/unet_relu_None_lr_0-001_bs_20_rmsprop")#unet_model_colab_3500i_30e_32bs_0005lr_relu.hdf5")
        img = self.read_in_rgb(path)
        img_resize = cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC)
        img_to_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
        img_norm = np.expand_dims(normalize(np.array(img_to_gray), axis=1), 2)#np.expand_dims(img_to_gray, 2)
        img_norm = img_norm[:, :, 0][:, :, None]
        to_input = np.expand_dims(img_norm, 0)
        prediction_mask = (unet.predict(to_input)[0, :, :, 0] > 0.2).astype(np.uint8)
        pred_mask = cv2.resize(prediction_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        k = self.mole_detect(img, pred_mask, self.delta)
        return k
    
    def remove_hair(self, image_rgb):
        # Convert the original image to grayscale
        grayScale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY )
        # Kernel for the morphological filtering
        kernel = cv2.getStructuringElement(1,(17,17))
        # Perform the blackHat filtering on the grayscale image to find the 
        # hair countours
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
        # intensify the hair countours in preparation for the inpainting 
        # algorithm
        _,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
        #print( thresh2.shape )
        # inpaint the original image depending on the mask
        dst = cv2.inpaint(image_rgb,thresh2,1,cv2.INPAINT_TELEA)
        return dst

    def cv_preprocessing(self, path):
        img = self.read_in_rgb(path)
        saturation = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 1]
        _, mask = cv2.threshold(saturation, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        mask = mask.astype(np.uint8)
        k = self.mole_detect(img, mask, self.delta)
        return k

