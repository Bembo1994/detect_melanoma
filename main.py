from dataset import dataset
from machine_learning import neural_network, svm
from utils import utils
from preprocessing import preprocess_image
from keras.models import load_model
from matplotlib import pyplot as plt
from tensorflow.keras.utils import normalize
import os, random
import cv2
import random
import numpy as np
import sys
import argparse

def main():
    '''
    preprocessor = preprocess_image.Preprocessor()
    for i in range(3):
        path = "dataset/ISICArchive/malignant/"+str(random.choice(os.listdir("dataset/ISICArchive/malignant/")))
        model = load_model("machine_learning/checkpoints/unet_model.hdf5")

        result2 = preprocessor.cv_preprocessing(path)
        result = preprocessor.unet_preprocessing(path, model)

        plt.figure(figsize=(16, 8))
        plt.subplot(231)
        plt.title('Original')
        plt.imshow(preprocessor.read_in_rgb(path))
        plt.subplot(232)
        plt.title('With UNet')
        plt.imshow(result)
        plt.subplot(233)
        plt.title('With CV')
        plt.imshow(result2)
        plt.show()

    '''

    ds = dataset.Dataset()

    ds.download_metadata()
    ds.download_dataset()
    ds.download_mask()

    ds.dataset_segmentation_to_pickle(0, utils.LIMIT_IMAGES_SEGMENTATION_PKL, False)
    ds.dataset_segmentation_to_pickle(utils.LIMIT_IMAGES_SEGMENTATION_PKL, utils.TEST_SET_SIZE, True)

    unet = neural_network.UNet()
    unet_model = unet.get_model(utils.IMG_SIZE_UNET, utils.IMG_SIZE_UNET, utils.IMG_CHANNELS_UNET)


    if eval(args.train_unet):
        X_train, X_val, y_train, y_val = ds.get_train_and_val_set_for_segmentation()
        unet.train(unet_model, X_train, X_val, y_train, y_val)
        _, acc = unet_model.evaluate(X_val, y_val)
        print("Accuracy = ", (acc * 100.0), "%")

    if eval(args.test_unet):
        _, X_test, _, y_test = ds.get_test_set_for_segmentation()
        # evaluate model
        _, acc = unet_model.evaluate(X_test, y_test)
        print("TEST Accuracy = ", (acc * 100.0), "%")


    '''
    X_train, X_test, y_train, y_test = ds.get_train_val_set_for_segmentation(k)

    unet_history = unet.train_and_get_history(unet_model, X_train, X_test, y_train, y_test, k)

    # evaluate model
    _, acc = unet_model.evaluate(X_test, y_test)
    print("Accuracy "+str(k)+"= ", (acc * 100.0), "%")

    # plot the training and validation accuracy and loss at each epoch
    loss = unet_history.history['loss']
    val_loss = unet_history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss'+str(k))
    plt.plot(epochs, val_loss, 'r', label='Validation loss'+str(k))
    plt.title('Training and validation loss'+str(k))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Fig_loss'+str(k)+'.png')

    acc = unet_history.history['accuracy']
    val_acc = unet_history.history['val_accuracy']

    plt.plot(epochs, acc, 'y', label='Training acc'+str(k))
    plt.plot(epochs, val_acc, 'r', label='Validation acc'+str(k))
    plt.title('Training and validation accuracy'+str(k))
    plt.xlabel('Epochs'+str(k))
    plt.ylabel('Accuracy'+str(k))
    plt.legend()
    plt.savefig('Fig_acc'+str(k)+'.png')


    unet_model = load_model("machine_learning/checkpoints/unet_model6.hdf5")


    ##################################
    # IOU
    X_train, X_test, y_train, y_test = ds.get_test_set_for_segmentation()
    y_pred = unet_model.predict(X_test)
    y_pred_thresholded = y_pred > 0.5

    intersection = np.logical_and(y_test, y_pred_thresholded)
    union = np.logical_or(y_test, y_pred_thresholded)
    iou_score = np.sum(intersection) / np.sum(union)
    print("IoU socre is: ", iou_score)

    #ds_train, ds_val = ds.get_train_val_set()

    #nn = neural_network.NN_VGG16()
    #nn.freeze_neural_network()
    #new_vgg16 = nn.add_top_layers()
    #nn.compile(new_vgg16)
    #history = nn.train(new_vgg16,ds_train,ds_val)
    #print(history)
    #new_vgg16.save('./my_model')


    #support_vector_machine = svm.SVM(dataset.Dataset().savePath)
    #support_vector_machine.load_dataset()
    #support_vector_machine.train()

    random_test_img = random.choice(os.listdir("dataset/ISICArchive/benign/"))
    #######################################################################
    # Predict on a few images
    #model = get_model()
    #unet_model.load_weights('mitochondria_test.hdf5')  # Trained for 50 epochs and then additional 100
    # model.load_weights('mitochondria_gpu_tf1.4.hdf5')  #Trained for 50 epochs

    test_img_number = random.randint(0, len(X_test))
    test_img = X_test[test_img_number]
    ground_truth = y_test[test_img_number]
    test_img_norm = test_img[:, :, 0][:, :, None]
    test_img_input = np.expand_dims(test_img_norm, 0)
    prediction = (unet_model.predict(test_img_input)[0, :, :, 0] > 0.2).astype(np.uint8)

    test_img_other = cv2.imread("dataset/ISICArchive/benign/" + random_test_img, 0)
    #cv2_imshow(test_img_other)
    test_img_other = cv2.resize(test_img_other, (256, 256), interpolation=cv2.INTER_CUBIC)
    # test_img_other = cv2.imread('data/test_images/img8.tif', 0)
    test_img_other_norm = np.expand_dims(normalize(np.array(test_img_other), axis=1), 2)
    test_img_other_norm = test_img_other_norm[:, :, 0][:, :, None]
    test_img_other_input = np.expand_dims(test_img_other_norm, 0)

    # Predict and threshold for values above 0.5 probability
    # Change the probability threshold to low value (e.g. 0.05) for watershed demo.
    prediction_other = (unet_model.predict(test_img_other_input)[0, :, :, 0] > 0.2).astype(np.uint8)

    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.title('External Image')
    plt.imshow(test_img_other, cmap='gray')
    plt.subplot(122)
    plt.title('Prediction of external Image')
    plt.imshow(prediction_other, cmap='gray')
    plt.show()
    '''
if __name__== "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_unet", help="train the unet ?")
    parser.add_argument("--test_unet", help="test the unet ?")
    args = parser.parse_args()
    main()