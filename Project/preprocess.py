import pandas as pd
import numpy as np
import torch
import src.utils as utils

from tqdm import tqdm

# general encoder class
class Encoder:
    def __init__(self, tokens: list) -> None:
        self.tokens = tokens
        self.enc_map = {}
        self.init_map()
    
    def encode_sequence(self, sequence: list):
        s = []
        for c in sequence:
            s.append(self.enc_map[c])
        return torch.Tensor(np.array(s))

    # sequences = [sequence]
    def encode_sequences(self, sequences: list):
        data = []
        for seq in sequences:
            data.append(self.encode_sequence(seq))
        return data

    def init_map(self):
        raise NotImplementedError

# encoder for LSTM
class OneHotEncoder(Encoder):
    def __init__(self, tokens: list) -> None:
        super().__init__(tokens)

    def init_map(self):
        length = len(self.tokens)
        for i, c in enumerate(self.tokens):
            self.enc_map[c] = np.eye(length)[:, i]

# encoder for Transformer
class NumericEncoder(Encoder):
    def __init__(self, tokens: list) -> None:
        super().__init__(tokens)
    
    # override super's definition
    def encode_sequence(self, sequence: list):
        s = []
        for c in sequence:
            s.append(torch.tensor(self.enc_map[c], dtype=torch.long))
        return torch.stack(s, dim=0)

    def init_map(self):
        for i, c in enumerate(self.tokens):
            self.enc_map[c] = i+1

def fix_training_data(df_train: pd.DataFrame):
    df = df_train.copy()
    df_train_updates = pd.read_csv("data/train_updates_20220929.csv", index_col="seq_id")

    all_features_nan = df_train_updates.isnull().all("columns")

    drop_indices = df_train_updates[all_features_nan].index
    df = df.drop(index=drop_indices)

    swap_ph_tm_indices = df_train_updates[~all_features_nan].index
    df.loc[swap_ph_tm_indices, ["pH", "tm"]] = df_train_updates.loc[swap_ph_tm_indices, ["pH", "tm"]]
    return df

def drop_nan_values(df: pd.DataFrame):
    return df.dropna(subset=["pH"])

def drop_long_sequences(df: pd.DataFrame, max_len: int):
    lengths = utils.get_sequence_lengths(df)
    return df[lengths <= max_len]

def zipped_encoded_data(df: pd.DataFrame, encoder: Encoder):
    idxs = utils.get_idxs(df)
    enc_sequences = encoder.encode_sequences(utils.get_sequences(df))
    phs = utils.get_phs(df)
    return list(zip(idxs, enc_sequences, phs))
