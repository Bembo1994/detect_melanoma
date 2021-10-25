from skimage.transform import resize
from skimage.io import imread
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
import pickle

class SVM():

    def __init__(self,save_path):
        self.categories = ['benign','malignant']
        self.flat_data_arr=[] #input array
        self.target_arr=[] #output array
        self.datadir=save_path
        self.param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.1, 1], 'kernel': ['rbf', 'poly']}
        self.svc = svm.SVC(probability=True)
        self.model = GridSearchCV(self.svc, self.param_grid)
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_dataset(self):
        #path which contains all the categories of images
        for i in self.categories:
          print(f'loading... category : {i}')
          path=os.path.join(self.datadir,i)
          for j,img in enumerate(os.listdir(path)):
            img_array=imread(os.path.join(path,img))
            img_resized=resize(img_array,(150,150,3))
            self.flat_data_arr.append(img_resized.flatten())
            self.target_arr.append(self.categories.index(i))
          print(f'loaded category:{i} successfully')
        flat_data=np.array(flat_data_arr)
        target=np.array(target_arr)
        df=pd.DataFrame(flat_data) #dataframe
        df['Target']=target
        x=df.iloc[:,:-1] #input data
        y=df.iloc[:,-1] #output data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
        print('Splitted Successfully')

    def train(self):
        self.model.fit(self.x_train, self.y_train)
        print('The Model is trained well with the given images')
        # model.best_params_ contains the best parameters obtained from GridSearchCV
        y_pred = self.model.predict(self.x_test)
        print("The predicted Data is :")
        print(y_pred)
        print("The actual data is:")
        print(np.array(y_test))
        print(f"The model is {accuracy_score(y_pred,self.y_test)*100}% accurate")
        # save the model to disk
        filename = 'svm.pkl'
        pickle.dump(self.model, open(filename, 'wb'))

