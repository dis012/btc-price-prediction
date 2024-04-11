import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import MSELoss
from src.model import MyDataset, BitCoinPredict
from notebooks.dataAnalysis import prepare_data, prepare_train_test, prepare_new_model_data, normalize_data, denormalize_data

print('Current directory: ', os.getcwd())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weights = torch.load('models/bitcoin_predict_weights.pth', map_location=device)

model = BitCoinPredict(1, 64, 2, dropout=0.2) # Can be adjusted to match the model architecture
model.load_state_dict(weights)
model.eval()

new_data = pd.read_csv('data/BTC-NEW.csv')
Data_new = new_data[['Date', 'Close']]
Data_new['Date'] = pd.to_datetime(Data_new['Date'])
Data_new = Data_new.set_index('Date')

Data_new, scaler_new = prepare_data(Data_new)
X, Y = prepare_new_model_data(Data_new)

with torch.no_grad():
    output = model(X)
    
output = output.flatten()
output_scale_back = np.zeros((X.shape[0], X.shape[1] + 1))
output_scale_back[:, 0] = output.numpy()  

# Denormalize
output_scale_back = denormalize_data(output_scale_back, scaler_new)

output_denorm = output_scale_back[:, 0].copy()

true_data = np.zeros((X.shape[0], X.shape[1] + 1))
true_data[:, 0] = Y.numpy().flatten()
true_data = denormalize_data(true_data, scaler_new)

# Plotting (adjusted to convert tensors to numpy if necessary)
plt.figure(figsize=(12, 6))
plt.plot(true_data[:,0], label='True Data')
plt.plot(output_denorm, label='Predicted Data')
plt.legend()
plt.show()
