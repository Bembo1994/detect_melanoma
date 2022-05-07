from dataset import dataset
from machine_learning import neural_network
from preprocessing import preprocess_image
import argparse
from xai import explainable_ai
from sklearn.metrics import confusion_matrix


def test_model(model, X_test, y_test, name_imgs, xai_abcd):
    if xai_abcd:
        preds = model.predict(X_test).ravel()
        for i in range(len(preds)):
            print("Pred {}\t Name {}".format(preds[i],name_imgs[i]))
    else:
        _, acc = model.evaluate(X_test, y_test)
        print("Accuracy (TEST) = ", (acc * 100.0), "%")
        if name_imgs != None:
            y_pred = model.predict(X_test).ravel()
            print(y_pred.min())
            print(y_pred.max())
            label = []
            pred = []
            misclassified_benign = []
            misclassified_malignant = []
            classified_right_benign = []
            classified_right_malignant = []
            for i, k in enumerate(y_pred):
                print("Predict: {}\tTrue_label: {}\tName_img: {}".format(k, y_test[i], name_imgs[i]))
                if k > 0.5:
                    label.append(y_test[i])
                    pred.append(1)
                    if y_test[i] == 0:
                        misclassified_benign.append(name_imgs[i])
                    else:
                        classified_right_malignant.append(name_imgs[i])
                else:
                    label.append(y_test[i])
                    pred.append(0)
                    if y_test[i] == 1:
                        misclassified_malignant.append(name_imgs[i])
                    else:
                        classified_right_benign.append(name_imgs[i])

            print("Misclassified Benign: \n{}\nMisclassified Malignant: \n{}".format(misclassified_benign, misclassified_malignant))
            conf = confusion_matrix(label, pred)
            print(conf)
            tn, fp, fn, tp = conf.ravel()
            print("TP {}\tTN {}\tFP {}\tFN {}".format(tp, tn, fp, fn))
            expl_ai = explainable_ai.Explainable_AI(model, misclassified_benign, misclassified_malignant, classified_right_benign, classified_right_malignant)
            expl_ai.get_saliency_and_gradcam_map()
            expl_ai.visualize_dense()
            expl_ai.visualize_conv()

def train_model(network, model, X_train, X_test, y_train, y_test, is_all_unfrozen):
    network.train(model, X_train, X_test, y_train, y_test, is_all_unfrozen)
    _, acc = model.evaluate(X_test, y_test)
    print("Accuracy (VALIDATION) = ", (acc * 100.0), "%")

def main():
    threshold_or_unet_preprocessing = args.threshold_or_unet_preprocessing
    ds = dataset.Dataset(threshold_or_unet_preprocessing)

    #ds.download_dataset_classification()
    #ds.download_dataset_segmentation()
    #ds.data_augmentation()
    '''
    train_networks = args.train_networks
    test_networks = args.test_networks
    xai_abcd = args.test_networks

    for net in train_networks:
        if net == "unet":
            X_train, X_test, y_train, y_test = ds.get_dataset_segmentation(False)
        else:
            X_train, X_test, y_train, y_test = ds.get_dataset_classification(net, False, False)
        network = neural_network.NeuralNetworks(net, threshold_or_unet_preprocessing)
        model = network.get_model()
        print("First training")
        train_model(network, model, X_train, X_test, y_train, y_test, False)
        if net != "unet":
            print("Training {} with layers unfrozen".format(net))
            train_model(network, model, X_train, X_test, y_train, y_test, True)
        if net in test_networks:
            if net == "unet":
                _, X_test, _, y_test = ds.get_dataset_segmentation(True)
                test_model(model, X_test, y_test, None, xai_abcd)
            else:
                X_test, y_test, name_imgs = ds.get_dataset_classification(net, True, False)
                test_model(model, X_test, y_test, name_imgs, False)

    for net in test_networks:
        network = neural_network.NeuralNetworks(net, threshold_or_unet_preprocessing)
        model = network.get_model()
        print(model.summary())
        if net == "unet":
            X_test, y_test = ds.get_dataset_segmentation(True)
            test_model(model, X_test, y_test, None, False)
        else:
            X_test, y_test, name_imgs = ds.get_dataset_classification(net, True, False)
            test_model(model, X_test, y_test, name_imgs, False)
        if xai_abcd :
            X_test, y_test, name_imgs = ds.get_dataset_classification(net, True, True)
            test_model(model, X_test, y_test, name_imgs, xai_abcd)
    '''
if __name__== "__main__" :
    parser = argparse.ArgumentParser()
    #parser.add_argument("--train_networks", nargs="+", default=[])
    #parser.add_argument("--test_networks", nargs="+", default=[])
    parser.add_argument("--threshold_or_unet_preprocessing", help="(type in 'threshold' or 'unet') Train neural networks with theshold or unet preprocessing?")
    #parser.add_argument("--xai_abcd", help="(True or False) explainable system with abcd properties?")
    args = parser.parse_args()
    main()