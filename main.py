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

    #shapes, train_test = ds.get_train_test_set_for_segmentation()
    train_test = ds.get_train_test_set_for_segmentation()

    X_train, X_test, y_train, y_test = train_test

    #h, w, c = shapes
    unet = neural_network.UNet()

    unet_model = unet.get_model(utils.IMG_SIZE_UNET, utils.IMG_SIZE_UNET, utils.IMG_CHANNELS_UNET)

    unet_history = unet.train_and_get_history(unet_model, X_train, X_test, y_train, y_test)

    print(unet_history)

    # evaluate model
    _, acc = unet_model.evaluate(X_test, y_test)
    print("Accuracy = ", (acc * 100.0), "%")


    # plot the training and validation accuracy and loss at each epoch
    loss = unet_history.history['loss']
    val_loss = unet_history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Fig21.png')

    acc = unet_history.history['accuracy']
    val_acc = unet_history.history['val_accuracy']

    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Fig22.png')

    ##################################
    # IOU

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
    '''

    '''
    test_img_number = random.randint(0, len(X_test))
    test_img = X_test[test_img_number]
    ground_truth = y_test[test_img_number]
    test_img_norm = test_img[:, :, 0][:, :, None]
    test_img_input = np.expand_dims(test_img_norm, 0)
    prediction = (unet_model.predict(test_img_input)[0, :, :, 0] > 0.2).astype(np.uint8)
    '''

    '''
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

if __name__== "__main__" :
    main()