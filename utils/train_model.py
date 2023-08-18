import math
import torch
import torch.nn as nn
import os
import pandas as pd

from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, lr_scheduler
from utils.schedulers import CosineScheduler

from models.transformers import CurveTransformer
from utils.dataset import load_data



def train(net, train_iter, valid_iter, test_iter, lr, num_epochs, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    #scheduler = CosineScheduler(60, warmup_steps=15, base_lr=lr, final_lr=0.1*lr)
    criterion = nn.MSELoss()

    # Create directory of saving models.
    if not os.path.isdir('../model_records'):
        os.mkdir('../model_records')

    best_loss, early_stop_count = math.inf, 0
    best_pred, best_y = [], []

    for epoch in range(num_epochs):
        net.train()
        loss_record, error_record = [], []

        for X, tk, y in train_iter:
            optimizer.zero_grad()
            X, tk, y = X.to(device), tk.to(device), y.to(device)
            pred = net(X, tk)
            loss = criterion(pred, y)
            error = torch.mean(torch.abs((pred - y) / y))
            loss.backward()
            clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            loss_record.append(loss.detach().item())
            error_record.append(error.detach().item())
        mean_train_loss = sum(loss_record) / len(loss_record)
        mean_train_error = sum(error_record) / len(error_record)

        mean_valid_loss, mean_valid_error, _, _ = valid_test(net, valid_iter, criterion, device)
        mean_test_loss, mean_test_error, pred_list, y_list = valid_test(net, test_iter, criterion, device)

        # Updata scheduler
        # if scheduler:
        #     if scheduler.__module__ == lr_scheduler.__name__:
        #         # UsingPyTorchIn-Builtscheduler
        #         scheduler.step()
        #     else:
        #         # Usingcustomdefinedscheduler
        #         for param_group in optimizer.param_groups:
        #             param_group['lr'] = scheduler(epoch)

        # Display current epoch number and loss
        print(f'Epoch [{epoch + 1}/{num_epochs}]: '
              f'Train loss: {mean_train_loss:.4f}, '
              f'Train error: {mean_train_error:.4f}, '
              f'Valid loss: {mean_valid_loss:.4f}, '
              f'Valid error: {mean_valid_error:.4f} ')

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(net.state_dict(), '../model_records/model.ckpt')
            print(f'Saving model with loss {best_loss:.4f} '
                  f'and error {mean_valid_error:.4f}')
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= 800:
            print('\nModel is not improving, so we halt the training session.')
            return




def valid_test(net, data_iter, criterion, device):
    net.eval()
    loss_record, error_record = [], []
    pred_list, y_list = [], []
    for X, tk, y in data_iter:
        X, tk, y = X.to(device), tk.to(device), y.to(device)
        with torch.no_grad():
            pred = net(X, tk)
            loss = criterion(pred, y)
            error = torch.mean(torch.abs((pred - y) / y))
            loss_record.append(loss.detach().item())
            error_record.append(error.detach().item())
            pred_list, y_list = pred.tolist(), y.tolist()
    mean_loss = sum(loss_record) / len(loss_record)
    mean_error = sum(error_record) / len(error_record)
    return mean_loss, mean_error, pred_list, y_list



if __name__ == "__main__":
    # test train
    train_iter, valid_iter, test_iter = load_data(batch_size=64, seed=12345678)
    # net = CurveTransformer(input_dim=2, embed_hidden_dim=4, d_model=32, ffw_hidden_dim=256,
    #                        pred_hidden_dim=128, num_heads=4, num_layers=4, dropout=0.1)
    net = CurveTransformer(input_dim=2, d_model=32, ffn_dim=128, pred_hidden_dim=16,
                           num_heads=8, num_layers=2, dropout=0.5)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Number of parameters: {total_params}")
    train(net, train_iter, valid_iter, test_iter, lr=1e-4, num_epochs=500, device=torch.device('cpu'))












