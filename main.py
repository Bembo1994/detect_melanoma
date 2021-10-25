from dataset import dataset
from machine_learning import neural_network, svm
import os
import statistics

def main():
    ds = dataset.Dataset()
    dataframe = ds.dataframe_analysis()
    print(dataframe.head())
    means = dataframe["mean"].dropna()
    print("STD of means : {}\n\n".format(statistics.stdev(means)))
    print("Mean of means : {}\n\n".format(sum(means)/len(means)))
    ds_train, ds_val = ds.get_train_val_set()

    nn = neural_network.NN_VGG16()
    nn.freeze_neural_network()
    new_vgg16 = nn.add_top_layers()
    nn.compile(new_vgg16)
    history = nn.train(new_vgg16,ds_train,ds_val)
    print(history)
    new_vgg16.save('./my_model')


    #support_vector_machine = svm.SVM(dataset.Dataset().savePath)
    #support_vector_machine.load_dataset()
    #support_vector_machine.train()


if __name__== "__main__" :
    main()