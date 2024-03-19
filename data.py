"""
This file contains all the custom Dataset classes of the approaches used
and a custom LightningDataModule class.
"""

import torch
from torch.utils.data import Dataset, DataLoader

from typing import Tuple

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import lightning as L


def tabular_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    Standartize a two dimension tensor by column (feature).

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Standartized tensor.
    """
    
    sigma, mu = torch.std_mean(tensor, dim=0)
    
    output = (tensor - mu)/sigma
    return output

class FF_Dataset(Dataset):
    def __init__(self, xez: Tuple[pd.core.frame.DataFrame],
                 super_am=False, p=1,
                 norm=False):
        
        self.args = ['xez', 'super_am', 'p', 'norm']
        
        self.z = torch.tensor(xez[2].values)
        
        if super_am:
            x = torch.tensor(xez[0].fillna(0).values)
            e = torch.tensor(xez[1].fillna(0).values)
            x = self.f.repeat(p, 1)
            e = self.e.repeat(p, 1)
            self.z = self.z.repeat(p, 1)
            self.covariables = torch.normal(x, e)
        else:
            self.covariables = torch.cat((torch.tensor(xez[0].fillna(0).values),
                                  torch.tensor(xez[1].fillna(0).values)), dim=1)
            
        if norm:
            self.covariables = tabular_norm(self.covariables)

    def __getitem__(self, idx):
        return self.covariables[idx], self.z[idx]

    def __len__(self):
        return len(self.z)
    
def make_vectors(x: torch.Tensor, e: torch.Tensor,
                 version=['no', 'with', 'stack']) -> torch.Tensor:

    n = len(x)
    p = len(x[0])
    
    x = torch.reshape(x, (n, 1, p))
    e = torch.reshape(e, (n, 1, p))

    if version == 'no':
        return x

    if version == 'with':
        return torch.cat((x, e), dim=2)
    if version == 'stack':
        return torch.cat((x, e), dim=1)
    
def matrix_1d_norm(tensor: torch.Tensor) -> torch.Tensor:
    sigma, mu = torch.std_mean(tensor, dim=(0, 2))
    
    n_channels = len(tensor[0])
    
    for i in range(n_channels):
        tensor[:, i, :] = (tensor[:, i, :] - mu[i])/sigma[i]
    return tensor
    
class CNN1D_Dataset(Dataset):
    def __init__(self, xez: Tuple[pd.core.frame.DataFrame],
                 version=['no', 'with', 'stack'],
                 super_am=False, p=1,
                 norm=False):

        if super_am and version != 'no':
            raise ValueError("super and not 'no'")

        self.z = torch.tensor(xez[2].values)

        x = torch.tensor(xez[0].fillna(0).values)
        e = torch.tensor(xez[1].fillna(0).values)

        if super_am:
            x = x.repeat(p, 1)
            e = e.repeat(p, 1)
            self.z = self.z.repeat(p, 1)
            new_x = torch.normal(x, e)
            self._1d = make_vectors(new_x, e, version='no')
        else:
            self._1d = make_vectors(x, e, version=version)
            
        if norm:
            self._1d = matrix_1d_norm(self._1d)

    def __getitem__(self, idx):
        return self._1d[idx], self.z[idx]

    def __len__(self):
        return len(self.z)
    

def make_matrices(
    x: torch.Tensor, e: torch.Tensor,
    K: int, t_inf: float, t_sup: float) -> torch.Tensor:
    n = len(x)
    p = len(x[0])
    
    xbar = x.nanmean(dim=1)
    xbar = xbar.reshape(n, 1, 1, 1)
    xbar = xbar.repeat(1, 1, K, p)
    
    x = x.reshape(n, 1, 1, p)
    x = torch.nan_to_num(x, float('inf'))
    x = x.repeat(1, 1, K, 1)
    
    e = e.reshape(n, 1, 1, p)
    e = torch.nan_to_num(e, 1e-6)
    e = e.repeat(1, 1, K, 1)
    
    t = -t_inf + (t_inf + t_sup)/(2*K) + (t_inf + t_sup)/(K)*torch.tensor(range(K))
    t = t.reshape(1, 1, K, 1)
    t = t.repeat(n, 1, 1, p)
    
    mu_p = t + xbar
    
    normal = torch.distributions.normal.Normal(loc=mu_p, scale=e)
    
    return normal.log_prob(x).exp()
    
class CNN2D_Dataset(Dataset): #add norm
    def __init__(self, xez: Tuple[pd.core.frame.DataFrame],
                 K: int, t_inf: float, t_sup: float,
                 norm=False):

        self.z = torch.tensor(xez[2].values)

        x = torch.tensor(xez[0].values)
        e = torch.tensor(xez[1].values)

        self._2d = make_matrices(x, e, K, t_inf, t_sup)

    def __getitem__(self, idx):
        return self._2d[idx], self.z[idx]

    def __len__(self):
        return len(self.z)


class Custom_DataModule(L.LightningDataModule):
    def __init__(self, *args, xez: Tuple[pd.core.frame.DataFrame],
                 dataset_class,
                 set_size=[0.5, 0.25, 0.25], seed: int=2023,
                 batch_size:int = None, num_workers: int=None,
                 **kwargs):
        super().__init__(*args)

        x_train, x_val = train_test_split(xez[0], test_size=(1 - set_size[0]),
                                          random_state=seed)
        e_train, e_val = train_test_split(xez[1], test_size=(1 - set_size[0]),
                                          random_state=seed)
        z_train, z_val = train_test_split(xez[2], test_size=(1 - set_size[0]),
                                          random_state=seed)

        if len(set_size) == 3:
            x_val, x_test = train_test_split(x_val, test_size=set_size[2] / (1 - set_size[0]),
                                             random_state=seed)
            e_val, e_test = train_test_split(e_val, test_size=set_size[2] / (1 - set_size[0]),
                                             random_state=seed)
            z_val, z_test = train_test_split(z_val, test_size=set_size[2] / (1 - set_size[0]),
                                             random_state=seed)
        else:
            x_test, e_test, z_test = x_val, e_val, z_val

        self.train = dataset_class((x_train, e_train, z_train), **kwargs)
        self.val = dataset_class((x_val, e_val, z_val), **kwargs)
        self.test = dataset_class((x_test, e_test, z_test), **kwargs)
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)

        

