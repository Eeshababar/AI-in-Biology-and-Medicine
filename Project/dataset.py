import torch
import numpy as np
import random

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

import src.utils

from src.preprocess import *
from src.config import *

class EnzymeDataset(Dataset):
    def __init__(self, X_zipped, y):       
        super(EnzymeDataset, self).__init__()
        self.X_zipped = X_zipped
        self.y = y

    def __len__(self):
        '''return len of dataset'''
        return len(self.X_zipped)

    def __getitem__(self, idx):
        '''return X, y, ph values at index idx'''
        i, x, ph = self.X_zipped[idx]
        
        if self.y is None:
            return i, x, None, ph
        else:
            return i, x, self.y[idx], ph

# pads per batch
def pad_collate_train(batch):
    idxs = [b[0] for b in batch]
    
    xs = [b[1] for b in batch]
    xs_pad = pad_sequence(xs, padding_value=0)
    
    ys = [b[2] for b in batch]
    ys = torch.Tensor(np.array(ys)).reshape(-1, 1)
        
    phs = [b[3] for b in batch]
    phs = torch.Tensor(np.array(phs)).reshape(-1, 1)
    return idxs, xs_pad, ys, phs

def pad_collate_test(batch):
    idxs = [b[0] for b in batch]
    
    xs = [b[1] for b in batch]
    xs_pad = pad_sequence(xs, padding_value=0)
        
    phs = [b[3] for b in batch]
    phs = torch.Tensor(np.array(phs)).reshape(-1, 1)
    return idxs, xs_pad, None, phs

def get_preprocessed_datasets(
    df_train_grouped: pd.DataFrame, 
    df_test: pd.DataFrame, 
    val_idxs: list,
    encoder: Encoder
):
    df_train_copy = df_train_grouped.copy(deep=True)
    df_test_copy = df_test.copy(deep=True)
    df_train_copy = drop_long_sequences(df_train_copy, MAX_LEN)

    # remove validation data
    for i in val_idxs:
        df_train_copy = df_train_copy[df_train_copy.iloc[:, 5] != i]

    Xz_train = zipped_encoded_data(df_train_copy, encoder)
    y_train = src.utils.get_tms(df_train_copy)
    Xz_test = src.preprocess.zipped_encoded_data(df_test_copy, encoder)

    train_dataset = EnzymeDataset(Xz_train, y_train)
    test_dataset = EnzymeDataset(Xz_test, None)
    
    return train_dataset, test_dataset

def get_preprocessed_grouped_datasets(
    df_train_grouped: pd.DataFrame, 
    encoder: Encoder, 
    test_size: float = 0.3, 
):
    df_group = df_train_grouped.copy(deep=True)
    df_group = drop_long_sequences(df_group, MAX_LEN)
    df_group = df_group[df_group.iloc[:, 5] != -1]
    groups = src.utils.get_group_list(df_group)

    datasets = []
    for i in groups:
        df_g = df_group[df_group.iloc[:, 5] == i]
        Xz = zipped_encoded_data(df_g, encoder)
        y = src.utils.get_tms(df_g)
        group_dataset = EnzymeDataset(Xz, y)
        datasets.append((i, group_dataset))
    
    size = 0
    total = len(df_group)
    val_datasets = []
    val_idxs = []
    
    while size / total <= test_size:
        i, dataset = random.sample(datasets, 1)[0]
        val_datasets.append(dataset)
        val_idxs.append(i)
        size += len(dataset)
        datasets.remove((i, dataset))

    train_datasets = [d for _, d in datasets]

    return train_datasets, val_datasets, val_idxs