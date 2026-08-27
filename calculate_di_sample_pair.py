import numpy as np
import os
# from mpi4py import MPI
from argparse import ArgumentParser
import torch

import pypardi.local_indices as li


# in this code, combine input and output together into a higher embedding space for calculating di

# install mpi4py to run parallel
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

# run serial
comm = None

class DISDataset(torch.utils.data.Dataset):
    def __init__(self, u, input_len, output_len, d=None):
        self.X, self.Y = [], []
        self.use_d = d is not None

        if self.use_d:
            self.d = []

        for i in range(len(u) - input_len - output_len):
            self.X.append(u[i:i+input_len])
            self.Y.append(u[i+input_len:i+input_len+output_len])
            if self.use_d:
                self.d.append(d[i+input_len])

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.float32)
        if self.use_d:
            self.d = torch.tensor(self.d, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.use_d:
            return self.X[idx], self.Y[idx], self.d[idx]
        else:
            return self.X[idx], self.Y[idx]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/workspace/git/DGNN/data/ks/train/data.npy') # kolmogorov, shallow_water, weather4degree
    parser.add_argument('--input_len', type=int, default=3)
    parser.add_argument('--output_len', type=int, default=1)
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--quantile', type=float, default=0.99)    
    # add custom save name
    parser.add_argument('--save_name', type=str, default=None, help='custom save name for d and theta')
    parser.add_argument('--save_dir', type=str, default=None, help='custom save directory for d and theta')
    args = parser.parse_args()

    data_path = args.data_path
    save_path = os.path.dirname(data_path)
    input_len = args.input_len
    output_len = args.output_len
    quantile = args.quantile
    n_samples = args.n_samples
    if args.save_name is not None:
        save_name = f'_sample_pair_in{input_len}_out{output_len}_q{quantile}_{args.save_name}_{n_samples}.npy'
    else:
        save_name = f'_sample_pair_in{input_len}_out{output_len}_q{quantile}_{n_samples}.npy'

    data = np.load(data_path) # [nt, nvar]
    data = data[:args.n_samples] if args.n_samples is not None else data
    data = data.reshape(data.shape[0],-1, 1) # [nt, n_space, nvar]
    dataset = DISDataset(data, input_len, output_len)
    X = dataset.X.numpy() # [n_samples, input_len, n_space, nvar]
    Y = dataset.Y.numpy() # [n_samples, output_len, n_space, nvar]
    # combine X and Y together for calculating di
    X = np.concatenate([X, Y], axis=1) # [n_samples, input_len+output_len, n_space, nvar]
    # reshape to [n_samples, (input_len+output_len)*n_space,nvar]
    n_samples, seq_len, n_space, nvar = X.shape
    X = X.reshape(n_samples, seq_len*n_space, nvar)
    

    results = li.compute(
        X=X, Y=None, ql=quantile, p=2, theta_fit="sueveges",
        distributed='none', p_value=None, dql=None,
        exp_test='anderson', comm=comm)

    d = results['d']
    theta = results['theta']

    # if rank == 0:
    #     np.save(os.path.join(save_path, 'd.npy'), d)
    #     np.save(os.path.join(save_path, 'theta.npy'), theta)
    #     print('mean d:', d.mean())
    #     print('mean theta:', theta.mean())
    if args.save_dir is not None:
        save_path = args.save_dir
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = os.path.dirname(data_path)
    # save_path = 'experiments/weather4degree/train_indices'  # weather 4 degree
    d = results['d']
    theta = results['theta']
    np.save(os.path.join(save_path, 'd'+save_name), d)
    np.save(os.path.join(save_path, 'theta'+save_name), theta)

    print('mean d:', np.nanmean(d))
    print('mean theta:', np.nanmean(theta))