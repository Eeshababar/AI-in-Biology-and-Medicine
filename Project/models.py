import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.config import *

class RNN(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        num_layers: int 
    ):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size=20, hidden_size=hidden_size, num_layers=num_layers)
        self.fc1 = nn.Linear(hidden_size+1, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, X, ph):
        # X.shape = [seq_len, batch_size, input_size]
        X = X.float()
        
        out, _ = self.lstm(X)
        out = out[-1, :, :] # take last output
        out = torch.cat((out, ph), 1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = MAX_LEN, batch_size: int = BATCH_SIZE):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.add(self.pe[:x.size(0)])
        return self.dropout(x)

class TransformerEncoderModel(nn.Module):
    def __init__(
        self,
        ntoken = NUM_TOKENS+1,
        d_model = 256, # embedding size
        nhead = 16,
        d_hid = 256,
        nlayers = 1,
        dropout = 0.5
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder = nn.Embedding(ntoken, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.d_model = d_model
        self.fc1 = nn.Linear(d_model+1, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, 1)

    def forward(self, src, ph):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
        """
        mask = torch.zeros(src.shape, device=src.device).masked_fill_(src > 0, 1).T
        
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        output = torch.mean(output, dim=0)
        output = torch.cat((output, ph), 1)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        return output