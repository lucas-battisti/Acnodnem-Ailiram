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


def tensor_norm(tensor: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, xez: Tuple[pd.Dataframe],
                 super_am=False, p=1, norm=False):
        
        self.args = ['xez', 'super_am', 'p', 'norm']
        
        self.z = torch.tensor(xez[2].values)
        
        if super_am:
            self.f = torch.tensor(xez[0].fillna(0).values)
            self.e = torch.tensor(xez[1].fillna(0).values)
            self.f = self.f.repeat(p, 1)
            self.e = self.e.repeat(p, 1)
            self.z = self.z.repeat(p, 1)
            self.cov = torch.normal(self.f, self.e)
        else:
            self.cov = torch.cat((torch.tensor(xez[0].fillna(0).values),
                                  torch.tensor(xez[1].fillna(0).values)), dim=1)
            
        if norm:
            self.cov = tensor_norm(self.cov)

    def __getitem__(self, idx):
        return self.cov[idx], self.z[idx]

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
    
class CNN1D_Dataset(Dataset):
    def __init__(self, xez: Tuple[pd.Dataframe],
                 version=['no', 'with', 'stack'],
                 super_am=False, p=1):

        if super_am and version != 'no':
            raise ValueError("super and not 'no'")

        self.z = torch.tensor(xez[2].values)

        x = torch.tensor(xez[0].fillna(0).values)
        e = torch.tensor(xez[1].fillna(0).values)

        if super_am:
            x = x.repeat(p, 1)
            e = e.repeat(p, 1)
            self.z = self.z.repeat(p, 1)
            self._1d = torch.normal(x, e)

        self._1d = make_vectors(x, e, version=version)

    def __getitem__(self, idx):
        return self._1d[idx], self.z[idx]

    def __len__(self):
        return len(self.z)
    
class Custom_DataModule(L.LightningDataModule):
    def __init__(self, *args, xez: Tuple[pd.Dataframe],
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

        

