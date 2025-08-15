import torch
import torch.nn as nn
import itertools
import numpy as np

from scipy import stats
from torch.utils.data import DataLoader

# warning: slow!
def spearman_score(
    data_loader: DataLoader, 
    model: nn.Module, 
    device: torch.device
):
    y_pred_list = []
    y_target_list = []
    
    model.eval()
    with torch.no_grad():
        for _, (_, X, y, ph) in enumerate(data_loader):
            X = X.to(device)
            y = y.to(device)
            ph = ph.to(device)
            output = model(X, ph)
            
            y_pred_list.append(output.reshape(-1).detach().cpu().numpy())
            y_target_list.append(y.reshape(-1).detach().cpu().numpy())
            
    y_pred_list = [a.tolist() for a in y_pred_list]
    y_target_list = [a.tolist() for a in y_target_list]
    y_pred = np.array(list(itertools.chain.from_iterable(y_pred_list)))
    y_target = np.array(list(itertools.chain.from_iterable(y_target_list)))
    model.train()
    
    score = stats.spearmanr(y_pred, y_target)[0]
    if np.isnan(score):
        return 0
    return score

def group_spearman_score(
    group_loaders: list,
    model: nn.Module,
    device: torch.device
):
    scores = np.array([spearman_score(loader, model, device) for loader in group_loaders])
    return scores.mean()

def generate_test_output(
    model: nn.Module, 
    data_loader: DataLoader, 
    filename: str,
    device: torch.device
):
    y_pred_list = []
    idx_list = []
    
    model.eval()
    with torch.no_grad():
        for i, (idx, X, _, ph) in enumerate(data_loader):
            X = X.to(device)
            ph = ph.to(device)
            output = model(X, ph)
            
            y_pred_list.append(output.reshape(-1).detach().cpu().numpy())
            idx_list.append(idx)
            
    y_pred_list = [a.tolist() for a in y_pred_list]
    y_pred = list(itertools.chain.from_iterable(y_pred_list))
    idxs = list(itertools.chain.from_iterable(idx_list))
    model.train()
    
    with open(filename, "w") as f:
        f.write("seq_id,tm\n")
        for i, tm in zip(idxs, y_pred):
            f.write(f"{i},{tm}\n")