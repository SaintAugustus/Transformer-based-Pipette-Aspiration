import torch

from models.transformers import CurveTransformer
from utils.configs import config
from utils.dataset import load_data
from utils.seeds import same_seed
from utils.train_model import train

seed = 0
same_seed(seed)
def main(lr, num_epochs, device, batch_size, input_dim, d_model, ffn_dim,
         pred_hidden_dim, num_heads, num_layers, dropout):
    train_iter, valid_iter, test_iter = load_data(batch_size=batch_size, seed=seed)
    net = CurveTransformer(input_dim=input_dim, d_model=d_model, ffn_dim=ffn_dim,
                           pred_hidden_dim=pred_hidden_dim, num_heads=num_heads,
                           num_layers=num_layers, dropout=dropout)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Number of parameters: {total_params}")
    train(net, train_iter, valid_iter, test_iter, lr=lr, num_epochs=num_epochs, device=torch.device(device))



if __name__ == '__main__':
    main(**config)
