import torch
import torch.nn as nn
import random

from torch.utils.data import DataLoader

from src.eval import group_spearman_score
from src.config import *

def train_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    train_group_loaders: list,
    train_group_loaders2: list,
    val_group_loaders: list, 
    num_epochs: int, 
    learning_rate: int,
    save_path: str,
    intra_group_loss: bool,
    device: torch.device
):
    losses = []
    scores = []

    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_score = -1e10
    model.train()
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (_, X, y, ph) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)
            ph = ph.to(device)
            
            # MSE loss
            output = model(X, ph)
            loss1 = criterion1(output, y)

            # intra-group loss
            if intra_group_loss:
                group_loader = random.sample(train_group_loaders, 1)[0]
                iterator = iter(group_loader)
                _, X1, y1, ph1 = next(iterator)
                _, X2, y2, ph2 = next(iterator)
                X1 = X1.to(device)
                X2 = X2.to(device)
                y1 = y1.to(device).reshape(1,1).float()
                y2 = y2.to(device).reshape(1,1).float()
                ph1 = ph1.to(device).reshape(1,1).float()
                ph2 = ph2.to(device).reshape(1,1).float()

                X1 = torch.swapaxes(X1, 0, 1)
                X2 = torch.swapaxes(X2, 0, 1)

                output1 = model(X1, ph1)
                output2 = model(X2, ph2)
                loss2 = criterion2(output1 - output2, y1 - y2)

                # Backward and optimize
                loss = (loss1 + loss2) / 2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append((loss1.item(), loss2.item(), loss.item()))
            else:
                loss = loss1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append((loss1.item(), 0, loss.item()))

            if (i+1) % 50 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, MSE Loss: {:4f}, Intra-group L1 Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item(), loss1.item(), loss2.item() if intra_group_loss else 0))

        train_group_score = group_spearman_score(train_group_loaders2, model, device)
        val_group_score = group_spearman_score(val_group_loaders, model, device)
        scores.append((train_group_score, val_group_score))

        print(f"Spearman coefficient on train dataset: {train_group_score}")
        print(f"Spearman coefficient on validation dataset: {val_group_score}")
        if val_group_score > best_score:
            best_score = val_group_score
            torch.save(model.state_dict(), save_path)

    return losses, scores