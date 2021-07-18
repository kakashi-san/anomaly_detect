import sys
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import pandas as pd
import os

class Data_Hanlder(object):
    
    def __init__(self,dataset_name,columns,time_steps):
        self.time_steps = time_steps

        
        self.data = pd.read_csv(dataset_name,index_col=0)
        self.columns = columns
        self.data[self.columns] = self.data[self.columns].shift(-1) - self.data[self.columns]
        self.data = self.data.dropna(how='any')
        self.pointer = 0
        self.train = np.array([])
        self.test = np.array([])
        self.test_label = np.array([])
        
        
        self.split_fraction = 0.2
        
        
    def _process_source_data(self):
 
        self._data_scale()
        self._data_arrage()
        self._split_save_data()
        
    def _data_scale(self):

        standscaler = StandardScaler()
        mscaler = MinMaxScaler(feature_range=(0,1))
        self.data[self.columns] = standscaler.fit_transform(self.data[self.columns])
        self.data[self.columns] = mscaler.fit_transform(self.data[self.columns])


    def _data_arrage(self):
        
        self.all_data = np.array([])
        self.labels = np.array([])
        d_array = self.data[self.columns].values  

        for index in range(self.data.shape[0]-self.time_steps+1):
            this_array = d_array[index:index+self.time_steps].reshape((-1,self.time_steps,len(self.columns)))
            if self.all_data.shape[0] == 0:
                self.all_data = this_array

            else:
                self.all_data = np.concatenate([self.all_data,this_array],axis=0)
        
    def _split_save_data(self):

        self.train = self.all_data
        np.save('train.npy',self.train)


    def _get_data(self):
        # if os.path.exists('/content/train.npy'):
        #     self.train = np.load('/content/train.npy')
        # if self.train.ndim ==3:
        #     if self.train.shape[1] == self.time_steps and self.train.shape[2] != len(self.columns):
        #         return 0
        self._process_source_data()


    def fetch_data(self,batch_size):
        if self.train.shape[0] == 0:
            self._get_data()
            
        if self.train.shape[0] < batch_size:
            return_train = self.train
        else:
            if (self.pointer + 1) * batch_size >= self.train.shape[0]-1:
                self.pointer = 0
                return_train = self.train[self.pointer * batch_size:,]
            else:
                self.pointer = self.pointer + 1
                return_train = self.train[self.pointer * batch_size:(self.pointer + 1) * batch_size,]
        if return_train.ndim < self.train.ndim:
            return_train = np.expand_dims(return_train,0)
        return return_train