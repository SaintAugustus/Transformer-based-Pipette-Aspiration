import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.dataset import load_data
from utils.train_model import train


class Net_3_layers(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_hidden3, n_output):
        super(Net_3_layers, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(nn.Linear(n_input, n_hidden1),
                                    nn.BatchNorm1d(n_hidden1), nn.Dropout(0.8))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden1, n_hidden2),
                                    nn.BatchNorm1d(n_hidden2), nn.Dropout(0.5))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden2, n_hidden3),
                                    nn.BatchNorm1d(n_hidden3), nn.Dropout(0.5))
        self.predict = nn.Linear(n_hidden3, n_output)

    def forward(self, x, tk):
        x = self.flatten(x)
        x = torch.cat((x, tk.unsqueeze(dim=1)), dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.predict(x)
        return x.reshape(-1)

class LiteDNN(nn.Module):
    def __init__(self, hid_dim1, hid_dim2, hid_dim3, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(2, 1)
        self.layers = nn.Sequential(
            nn.Linear(2, hid_dim1),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hid_dim1, hid_dim2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hid_dim2, hid_dim3),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hid_dim3, 1)
        )

    def forward(self, X, tk):
        X = F.relu(self.dense(X))
        X = X.squeeze().mean(dim=1).reshape(-1, 1)
        tk = tk.unsqueeze(dim=1)
        out = torch.cat((X, tk), dim=1)
        return self.layers(out).reshape(-1)



if __name__ == "__main__":
    # load data
    train_iter, _, _ = load_data(batch_size=64, seed=0)
    batch = next(iter(train_iter))
    print(batch[0].shape, batch[1].shape, batch[2].shape)

    X, tk, y = batch

    # # test dnn
    # net = Net_3_layers(257, 1024, 128, 32, 1)
    # pred = net(X, tk)
    # print(pred.shape)

    # test litednn
    net = LiteDNN(64, 16, 8)
    pred = net(X, tk)
    print(pred.shape)

    # train litednn
    train_iter, valid_iter, test_iter = load_data(batch_size=64, seed=123456)
    train(net, train_iter, valid_iter, lr=1e-4, num_epochs=1000, device=torch.device('cpu'))


