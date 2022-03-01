from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.scores import BinaryScore, CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear, ExtractIntermediateLayer
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam
from matplotlib import cm
import os
from dataset import dataset
import random
from preprocessing import preprocess_image
import numpy as np
from matplotlib import pyplot as plt
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.activation_maximization.callbacks import Progress
from tf_keras_vis.activation_maximization.input_modifiers import Jitter, Rotate2D, Scale
from tf_keras_vis.activation_maximization.regularizers import Norm, TotalVariation2D
import tensorflow as tf

class Explainable_AI():

    def __init__(self, model):
        self.real_path = os.path.dirname(os.path.realpath(__file__))
        self.model = model
        self.replace2linear = ReplaceToLinear()
        self.preprocessor = preprocess_image.Preprocessor()
        self.preprocessor.size = (299, 299)

    def plot(self, smoothgrad_saliency_map, images, titles): #in first pos is benign, second is malignant
        f, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        axs[0, 0].set_title("{}".format(titles[0]), fontsize=10)
        axs[0, 0].imshow(images[0])
        axs[0, 0].axis('off')

        axs[0, 1].set_title("SmoothGrad Saliency Map", fontsize=10)
        axs[0, 1].imshow(smoothgrad_saliency_map[0], cmap='jet')
        axs[0, 1].axis('off')

        plt.tight_layout()
        plt.savefig("smoothgrad_map.png")
        plt.show()

    def get_saliencymap(self, image):
        score = BinaryScore([0,1])

        saliency = Saliency(self.model, model_modifier=self.replace2linear, clone=True)

        images = np.asarray([np.array(self.benign_img), np.array(self.malignant_img)])
        titles = [self.random_img_benign.split(".")[0],self.random_img_malignant.split(".")[0]]

        smoothgrad_saliency_map = saliency(score, images, smooth_samples=20, smooth_noise=0.20) # The number of calculating gradients iterations. # noise spread level.

        self.plot(smoothgrad_saliency_map, images, titles)

        titles = [self.true_benign.split(".")[0],self.true_malignant.split(".")[0]]

        images = np.asarray([np.array(self.true_benign_img), np.array(self.true_malignant_img)])

        smoothgrad_saliency_map = saliency(score, images, smooth_samples=20, smooth_noise=0.20) # The number of calculating gradients iterations. # noise spread level.
        self.plot(vanilla_saliency_map, smoothgrad_saliency_map, cam_map, cam_pp_map, scorecam_map, images, titles)


    def visualize_dense(self):
        score = BinaryScore([0])
        # Generate maximized activation
        activation_maximization = ActivationMaximization(self.model, model_modifier=self.replace2linear, clone=True)
        activations = activation_maximization(score, callbacks=[Progress()])

        # Render
        f, ax = plt.subplots(figsize=(4, 4))
        for i, e in enumerate(activations):
            ax.imshow(activations[i])
            if i%2==0:
                ax.set_title('Benign', fontsize=16)
            else:
                ax.set_title('Malignant', fontsize=16)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def visualize_conv(self):
        layer_name = 'conv2d_93'
        extract_intermediate_layer = ExtractIntermediateLayer(index_or_name=layer_name)
        #filter_number = 0
        score = BinaryScore([0])#CategoricalScore(filter_number)
        activation_maximization = ActivationMaximization(self.model, model_modifier=[extract_intermediate_layer, self.replace2linear], clone=True)
        activations = activation_maximization(score, callbacks=[Progress()])

        # Render
        f, ax = plt.subplots(figsize=(4, 4))
        for i, e in enumerate(activations):
            ax.imshow(activations[i])
            if i%2==0:
                ax.set_title('Benign', fontsize=16)
            else:
                ax.set_title('Malignant', fontsize=16)
            ax.axis('off')
        plt.tight_layout()
        plt.show()