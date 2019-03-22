#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 20:40:55 2019

@author: ag
"""
from model import Model
import matplotlib.pyplot as plt
from dataset import Data
from sklearn.metrics import mean_absolute_error
import pandas as pd
from pathlib import Path
import os

def main():
    p=Path(__file__).parents[0]
    directory = os.path.abspath(os.path.join(p,"gemini_ETHUSD_d.csv"))
    
    data = Data(directory)
    train,test = data.split_data(test_size=0.2)
    numOfDays=20
    x_train,x_test,y_train,y_test = data.prepare_data(train,test,numOfDays)
    
    
    model = Model()   
    #hyperparameters tuning
    epochs = 50
    optimizer='adam'
    loss='mean_squared_error'
    activation ='tanh'
    batch_size = 1
    neurons = 30
    
    
    model.LSTM_model(x_train,activation =activation,optimizer=optimizer,loss=loss,neurons=neurons)
    history = model.train(x_train,y_train,x_test,y_test,epochs=epochs,batch_size=batch_size)



    targets = test['Close'][numOfDays:]
    preds = model.predict(x_test).squeeze()
    
    print('MAE: ',mean_absolute_error(preds,y_test))
    
    
    preds = test['Close'].values[:-numOfDays] * (preds + 1)
    preds = pd.Series(index=targets.index, data=preds)
    
    line_plot(targets, preds, 'actual', 'prediction', lw=3)
    line_plot(history.history['loss'],history.history['val_loss'],'train loss','test loss',lw=3)



def line_plot(line1, line2, label1=None, label2=None, title='ETH', lw=2):
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('price', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=18)
    plt.show()
    
    
if __name__ =='__main__':
    main()
