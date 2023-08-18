import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer

from utils.dataset import load_data


class PositionalEncoding(nn.Module):
    """
    X, P shape (n, d);
    theta = i / 10000^(2j/d)
    P[i, 2j] = sin(theta), P[i, 2j + 1] = cos(theta)
    """
    def __init__(self, hidden_dim, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, hidden_dim))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
        Y = torch.pow(10000, torch.arange(0, hidden_dim, 2, dtype=torch.float32) / hidden_dim)
        theta = X / Y
        self.P[:, :, 0::2] = torch.sin(theta)
        self.P[:, :, 1::2] = torch.cos(theta)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class CurveEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 dropout, seq_len=128, **kwargs):
        super().__init__(**kwargs)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.layer1 = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm2d(1)
        self.ln = nn.LayerNorm((seq_len, 2))

    def forward(self, X):
        # X = X.unsqueeze(dim=1)
        # X = self.bn(X).squeeze(1)
        X = self.ln(X)
        return self.layers(X)

class DenseBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )

    def forward(self, X):
        Y = self.layers(X)
        X = torch.cat((X, Y), dim=2)
        return X

class DenseCurveEmbedding(nn.Module):
    def __init__(self, input_dim, num_blks, dropout, seq_len=128, **kwargs):
        super().__init__(**kwargs)
        self.ln = nn.LayerNorm((seq_len, input_dim))
        self.mlp_layer = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim),
            nn.ReLU()
        )
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("dense_embed_" + str(i),
                    DenseBlock(2 * input_dim, 2 * input_dim, dropout))

    def forward(self, X):
        b, l = X.shape[0], X.shape[1]
        X = self.ln(X)
        X = self.mlp_layer(X)
        # X shape (b, l, d)
        X = X.unsqueeze(dim=2)
        # X shape (b, l, c, d)
        for blk in self.blks:
            X = blk(X)
        X = X.reshape((b, l, -1))
        return X

class CnnEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv1d(in_channels, out_channels//4, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(out_channels//4, out_channels//2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(out_channels//2, out_channels, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm1d(out_channels//4)
        self.bn2 = nn.BatchNorm1d(out_channels//2)
        self.bn3 = nn.BatchNorm1d(out_channels)

    def forward(self, X):
        Y = F.gelu(self.bn1(self.conv1(X.permute(0, 2, 1))))
        Y = F.gelu(self.bn2(self.conv2(Y)))
        Y = F.gelu(self.bn3(self.conv3(Y)))
        return Y.permute(0, 2, 1)

class PredNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, X):
        return self.layers(X)

class TkNet(nn.Module):
    def __init__(self, y_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.layers = nn.Sequential(
            nn.Linear(1, y_dim),
            nn.Dropout(dropout),
            nn.GELU()
        )

    def forward(self, X):
        Y = self.layers(X)
        out = torch.cat((Y, X), dim=1)
        return out


class CurveTransformer(nn.Module):
    def __init__(self, input_dim, d_model, ffn_dim, pred_hidden_dim,
                 num_heads, num_layers, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        # self.embedding = CurveEmbedding(input_dim, embed_hidden_dim, d_model, dropout)
        # self.embedding = DenseCurveEmbedding(input_dim, num_blks, dropout)
        self.embedding = CnnEmbedding(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("att_block_" + str(i),
                TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                        dim_feedforward=ffn_dim, dropout=dropout,
                                        activation=F.gelu))
        self.dense = nn.Linear(d_model, 1)
        self.tk_net = TkNet(y_dim=d_model//4, dropout=dropout)
        self.prednet = PredNet(d_model + d_model//4 + 1, pred_hidden_dim, dropout)

    def forward(self, X, tk):
        X = self.embedding(X)
        X = self.pos_encoding(X * math.sqrt(self.d_model))
        X = X.permute(1, 0, 2) # length first
        for blk in self.blks:
            X = blk(X, None)
        X = X.permute(1, 0, 2) # batch first
        tk = tk.unsqueeze(1)
        new_tk = self.tk_net(tk)
        # pooling
        new_X = F.softmax(self.dense(X).permute(0, 2, 1), dim=2)
        new_X = torch.bmm(new_X, X).squeeze()
        # concat new_X and tk
        new_X = torch.cat((new_X, new_tk), dim=1)
        pred = self.prednet(new_X)
        return pred.reshape(-1)




if __name__ == "__main__":
    # load data
    train_iter, _, _ = load_data(batch_size=64, seed=0)
    batch = next(iter(train_iter))
    print(batch[0].shape, batch[1].shape, batch[2].shape)

    X, tk, y = batch

    # # test CurveEmbedding
    # net = CurveEmbedding(input_dim=2, hidden_dim=8, output_dim=32, dropout=0.1)
    # pred = net(X)
    # print(pred.shape)

    # test ResCurveEmbedding
    # net = DenseCurveEmbedding(input_dim=1, num_blks=2, dropout=0.1)
    # pred = net(X)
    # print(pred.shape)

    # test CnnEmbedding
    net = CnnEmbedding(in_channels=2, out_channels=32)
    pred = net(X)
    print(pred.shape)

    # test CurveTransformer
    net = CurveTransformer(input_dim=2, d_model=32, ffn_dim=128, pred_hidden_dim=8,
                           num_heads=4, num_layers=2, dropout=0.5)
    pred = net(X, tk)
    print(pred.shape)




