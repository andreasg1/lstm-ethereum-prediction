#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:26:50 2019

@author: ag
"""
import datetime as dt
from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential

class Model():
    
    def __init__(self):
        self.model = Sequential()


    def LSTM_model(self,x_train,activation,optimizer,loss,neurons):
       
        self.model.add(LSTM(units=neurons,activation = activation,return_sequences = True,input_shape=(x_train.shape[1],x_train.shape[2])))
        self.model.add(Dropout(0.25))
        self.model.add(LSTM(units=neurons,activation=activation))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(units=1))
       
        
        self.model.compile(loss=loss,optimizer=optimizer)
        
    
    
    def train(self,x_train,y_train,x_test,y_test,epochs,batch_size):
        
        start = dt.datetime.now()
        history = self.model.fit(x= x_train,y=y_train,validation_data=(x_test, y_test),batch_size=batch_size,epochs=epochs,shuffle=False)
        end = dt.datetime.now()
        print('Time taken: ',(end-start))
        
        return history
    
    def predict(self,x_test):
        preds = self.model.predict(x_test).squeeze()
        
        return  preds