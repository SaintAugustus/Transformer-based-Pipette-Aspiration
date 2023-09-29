import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

#import data
def get_data(df_x, df_y):
    x, y, tk, tgt = df_x.values[:, 1:], df_y.values[:, 1: -3], \
                    df_y.values[:, -3], df_y.values[:, -2]
    x, y, tk, tgt = torch.tensor(x, dtype=torch.float32), \
                    torch.tensor(y, dtype=torch.float32), \
                    torch.tensor(tk, dtype=torch.float32), \
                    torch.tensor(tgt, dtype=torch.float32)
    x, y = torch.unsqueeze(x, dim=2), torch.unsqueeze(y, dim=2)
    culve = torch.cat((x, y), dim=2)

    return culve, tk, tgt

def train_valid_test_split(data_set, test_ratio, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    test_set_size = int(test_ratio * len(data_set))
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size - test_set_size
    train_valid_set, test_set = random_split(data_set, [train_set_size + valid_set_size, test_set_size], generator=torch.Generator().manual_seed(seed))
    train_set, valid_set = random_split(train_valid_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))

    # pd_train = pd.DataFrame(train_set.dataset.dataset, index=train_set.indices)
    # pd_valid = pd.DataFrame(valid_set.dataset.dataset, index=valid_set.indices)
    # pd_test = pd.DataFrame(test_set.dataset, index=test_set.indices)
    return train_set.indices, valid_set.indices, test_set.indices

# pipe data set
class PipeDataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, culve, tk, tgt):
        self.culve = culve
        self.tk = tk
        self.tgt = tgt

    def __getitem__(self, idx):
        return self.culve[idx], self.tk[idx], self.tgt[idx]

    def __len__(self):
        return len(self.culve)

def load_data(batch_size, seed, seq_len=256):
    df_x = pd.read_csv(f'../dataset/x-dataset-{seq_len}.csv')
    df_y = pd.read_csv(f'../dataset/y-dataset-{seq_len}.csv')

    # split data
    train_id, valid_id, test_id = train_valid_test_split(df_x, 0.1, 0.1, seed=seed)
    train_x, train_y = pd.DataFrame(df_x, index=train_id), pd.DataFrame(df_y, index=train_id)
    valid_x, valid_y = pd.DataFrame(df_x, index=valid_id), pd.DataFrame(df_y, index=valid_id)
    test_x, test_y = pd.DataFrame(df_x, index=test_id), pd.DataFrame(df_y, index=test_id)

    # get tensor data
    train_culve, train_tk, train_tgt = get_data(train_x, train_y)
    valid_culve, valid_tk, valid_tgt = get_data(valid_x, valid_y)
    test_culve, test_tk, test_tgt = get_data(test_x, test_y)

    # get iter, valid and test use the whole batch
    train_iter = DataLoader(PipeDataset(train_culve, train_tk, train_tgt), batch_size=batch_size, shuffle=True)
    valid_iter = DataLoader(PipeDataset(valid_culve, valid_tk, valid_tgt), batch_size=1000, shuffle=False)
    test_iter = DataLoader(PipeDataset(test_culve, test_tk, test_tgt), batch_size=1000, shuffle=False)

    return train_iter, valid_iter, test_iter


if __name__ == "__main__":
    # test get_data
    df_x = pd.read_csv('../dataset/x-dataset.csv')
    df_y = pd.read_csv('../dataset/y-dataset.csv')
    culve, tk, tgt = get_data(df_x, df_y)

    # test train_valid_test_split
    train_id, valid_id, test_id = train_valid_test_split(df_x, 0.1, 0.1, seed=0)
    print(len(train_id) / len(df_x), len(valid_id) / len(df_x), len(test_id) / len(df_x))

    # test load_data
    train_iter, valid_iter, test_iter = load_data(batch_size=64, seed=0)
    batch = next(iter(train_iter))
    print(batch[0].shape, batch[1].shape, batch[2].shape)











