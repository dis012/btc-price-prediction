import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
class BitCoinPredict(nn.Module):
    def __init__(self, input_size, hidden_size, num_stack, dropout=0.5):
        super(BitCoinPredict, self).__init__()
        self.hidden_size = hidden_size
        self.num_stack = num_stack
        self.input_size = input_size  
        self.lstm = nn.LSTM(input_size, hidden_size, num_stack, batch_first=True, dropout=dropout if num_stack > 1 else 0)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stack, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_stack, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.layer_norm(out[:, -1, :])
        out = self.fc(out)
        out = self.sigmoid(out)
        return out