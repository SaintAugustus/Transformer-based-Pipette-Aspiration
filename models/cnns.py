import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.dataset import load_data
from utils.train_model import train


class CNNRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(2, 1)
        self.hidden1 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=16, padding=4, stride=8),
            nn.BatchNorm1d(4))
        self.hidden2 = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=8, padding=2, stride=4),
            nn.BatchNorm1d(8))
        self.linear1 = nn.Sequential(
            nn.Linear(65, 32),
            nn.BatchNorm1d(32), torch.nn.Dropout(0.5))
        self.linear2 = nn.Sequential(
            nn.Linear(32, 8),
            nn.BatchNorm1d(8), nn.Dropout(0.5))
        self.predict = nn.Linear(8, 1)

    def forward(self, x, tk):
        b = x.shape[0]
        x = self.dense(x) # (b, l, 1)
        x = F.relu(x)
        x = x.permute(0, 2, 1)
        # conv
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        # cat
        x = x.reshape(b, -1)
        tk = tk.unsqueeze(dim=1)
        x = torch.cat((x, tk), 1)
        # dnn
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.predict(x)
        return x.reshape(-1)

if __name__ == "__main__":
    # load data
    train_iter, _, _ = load_data(batch_size=64, seed=0)
    batch = next(iter(train_iter))
    print(batch[0].shape, batch[1].shape, batch[2].shape)

    X, tk, y = batch

    # test cnn
    net = CNNRegression()
    pred = net(X, tk)
    print(pred.shape)

    # train cnn
    train_iter, valid_iter, test_iter = load_data(batch_size=64, seed=88888888)
    train(net, train_iter, valid_iter, lr=0.001, num_epochs=1000, device=torch.device('cpu'))






