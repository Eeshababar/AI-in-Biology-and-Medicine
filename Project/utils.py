import pandas as pd
import numpy as np

def get_sequence_lengths(df: pd.DataFrame):
    sequences = get_sequences(df)
    return np.array([len(s) for s in sequences])

def get_sequences(df: pd.DataFrame):
    return df.iloc[:,0].to_list()

def get_phs(df: pd.DataFrame):
    return df.iloc[:, 1].to_list()

# for train only; test data do not have tms
def get_tms(df: pd.DataFrame):
    return df.iloc[:,3].to_list()

def get_idxs(df: pd.DataFrame):
    return df.index.to_list()

def get_group_list(df_group: pd.DataFrame):
    groups = df_group.iloc[:, 5].unique()
    groups = groups[groups != -1]
    return groups.tolist()

def save_losses(losses: list, filename: str):
    with open(filename, "w") as f:
        f.write("MSE loss,Intra-group L1 loss,Total loss\n")
        for l1, l2, l in losses:
            f.write(f"{l1},{l2},{l}\n")

def save_scores(scores: list, filename: str):
    with open(filename, "w") as f:
        f.write("Train score, Val score\n")
        for train, val in scores:
            f.write(f"{train},{val}\n")