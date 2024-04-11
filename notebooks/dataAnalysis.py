import pandas as np
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

def normalize_data(iData):
    oData = iData.copy()
    scaler = MinMaxScaler()
    oData = scaler.fit_transform(oData)
    
    return oData, scaler

def prepare_data(iData):
    oData = iData.copy()
    
    for i in range(1, 6):
        oData[f'Close-{i}'] = iData['Close'].shift(i)
    
    oData = oData.dropna()
    oData = np.array(oData)
    oData, scaler = normalize_data(oData)  # Adjust to capture scaler
    
    return oData, scaler

def denormalize_data(iData, scaler):
    oData = scaler.inverse_transform(iData)
    return oData

def prepare_train_test(iData):
    idx = int(len(iData) * 0.9)
    
    y = iData[:, 0]
    x = np.flip(iData[:, 1:], axis=1).copy()
    
    Y_train = torch.tensor(y[:idx]).float()
    X_train = torch.tensor(x[:idx]).float()
    
    Y_test = torch.tensor(y[idx:]).float()
    X_test = torch.tensor(x[idx:]).float()
    
    Y_train = torch.reshape(Y_train, (Y_train.shape[0], 1))
    Y_test = torch.reshape(Y_test, (Y_test.shape[0], 1))
    
    X_train = torch.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = torch.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    return X_train, Y_train, X_test, Y_test

def prepare_new_model_data(iData):
    data = iData.copy()
    Y = torch.tensor(data[:, 0]).float()
    X = np.flip(data[:, 1:], axis=1).copy()
    X = torch.tensor(X).float()
    
    Y = torch.reshape(Y, (Y.shape[0], 1))
    X = torch.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, Y