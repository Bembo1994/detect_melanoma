from dataset import dataset
import os
import statistics


def main():
    ds = dataset.Dataset()
    dataframe = ds.dataframe_analysis()
    print(dataframe.head())
    means = dataframe["mean"].dropna()
    print("STD of means : {}\n\n".format(statistics.stdev(means)))
    print("Mean of means : {}\n\n".format(sum(means)/len(means)))
    ds.dataset_analysis()
if __name__== "__main__" :
    main()