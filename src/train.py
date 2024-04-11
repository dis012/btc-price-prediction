import os
import sys

current_script_path = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_script_path)
sys.path.append(parent_directory)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import MSELoss
from model import MyDataset, BitCoinPredict
from notebooks.dataAnalysis import prepare_data, prepare_train_test, prepare_new_model_data, normalize_data, denormalize_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cuda':
    print('Using GPU')
else:
    print('Using CPU')

df = pd.read_csv('BTC-USD.csv')
data = df[['Date', 'Close']]

data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

Data, scaler = prepare_data(data)

X_train, Y_train, X_test, Y_test = prepare_train_test(Data)

train_dataset = MyDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = MyDataset(X_test, Y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = BitCoinPredict(1, 64, 2, dropout=0.2) # input_size, hidden_size, num_stack 

learning_rate = 0.001
num_epochs = 100

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

def train():
    model.train()
    total_loss = 0
    
    for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, Y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss {loss.item()}')

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch Average Loss: {avg_loss}')

 
def validate():
    model.eval() # Set model to evaluation mode
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            output = model(X_batch)
            loss = criterion(output, Y_batch)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    print(f'Validation Avg Loss: {avg_loss}')
    
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    train()
    validate()
    scheduler.step()
    
models_dir = os.path.join(parent_directory, 'models')
os.makedirs(models_dir, exist_ok=True)

model_weights_path = os.path.join(models_dir, 'bitcoin_predict_weights.pth')
torch.save(model.state_dict(), model_weights_path)

# Testing
with torch.no_grad():
    output = model(X_test)
    
output = output.flatten()
output_scale_back = np.zeros((X_test.shape[0], X_test.shape[1] + 1))
output_scale_back[:, 0] = output.numpy()  

# Denormalize
output_scale_back = denormalize_data(output_scale_back, scaler)

output_denorm = output_scale_back[:, 0].copy()

true_data = np.zeros((X_test.shape[0], X_test.shape[1] + 1))
true_data[:, 0] = Y_test.numpy().flatten()
true_data = denormalize_data(true_data, scaler)

# Plotting 
plt.figure(figsize=(12, 6))
plt.plot(true_data[:,0], label='True Data')
plt.plot(output_denorm, label='Predicted Data')
plt.legend()
plt.show()