import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# =============================================================================
# https://www.cryptodatadownload.com
# Daily dataset
# gemini_ETHUSD_d.csv
# =============================================================================
class Data():
    
    def __init__(self,filename):
        self.data = pd.read_csv(filename,engine = 'python')
         #remove timestamp and symbol feature
        data = self.data
        data = data.drop('Volume USD',axis=1)
        data = data.drop('Symbol',axis=1)
        
        #reverse order by date
        data = data.iloc[::-1]
        data = data.set_index('Date')
        self.data = data
        #split data into train and test set
        
        
    
    def split_data(self,test_size):
        split_point= len(self.data)- int(test_size * len(self.data))
        train = self.data.iloc[:split_point]
        test = self.data.iloc[split_point:]
        
        return train,test
    
        
    def plot_closeFeature(self,data):
        plt.figure(figsize = (18,9))
        plt.plot(range(data.shape[0]),(data['Close']))
        plt.xticks(range(0,data.shape[0],1000),data['Date'].loc[:: 1000],rotation=45)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Close Price',fontsize=18)
        plt.show()
    
    def dataset_exploration(self,data):
        print(data.head())
        print('Dataset contains null values: ',data.isnull().values.any())
        print('Dataset length: ',len(data))
        print('Dataset shape: ',data.shape)
        print('\nDataset statistics:\n')
        print(data.describe())
        print(data.columns)
        
    #split to time series based on days
    def split_to_timeSeries(self,data,numOfDays):
        
        time_series = []
        for index in range(len(data)-numOfDays):
            time_series.append(data[index:(index+numOfDays)])
            
        return time_series 
    
    
    def prepare_data(self,train,test,windowLength):
        
        scaler  = MinMaxScaler(feature_range=(0,1))
        #normalize datasets using min max in range 0 and 1
        training = scaler.fit_transform(train.values)
        testing = scaler.transform(test.values)
        
        #extract window data -> features
        train_series = self.split_to_timeSeries(training,windowLength)
        x_train = np.array(train_series)
        
        #same for test set
        test_series = self.split_to_timeSeries(testing, windowLength)      
        x_test = np.array(test_series)


        #extract target data-> labels
        y_train = train['Close'][windowLength:].values
        y_train = np.reshape(y_train,(len(y_train),1))
        
        y_test = test['Close'][windowLength:].values
        y_test = np.reshape(y_test,(len(y_test),1))
        
        #normalize labels
        Y_train = np.zeros(shape=(len(y_train), 5) )
        Y_train[:,0] = y_train[:,0]
        y_train = scaler.transform(Y_train)[:,0]
        
        
        Y_test = np.zeros(shape=(len(y_test), 5) )
        Y_test[:,0] = y_test[:,0]
        y_test = scaler.transform(Y_test)[:,0]
    
    
        return x_train,x_test,y_train,y_test
    

    
    
