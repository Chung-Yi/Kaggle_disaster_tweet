import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, split_weight, num_workers):
        self.n_samples = len(dataset)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': False,
            'pin_memory' : True,
            # 'collate_fn': collate_fn,
            'num_workers': num_workers
        }

        self.train_sampler, self.val_sampler, self.test_sampler = self._split_sampler(split_weight)

        super().__init__(**self.init_kwargs)


    def _split_sampler(self, split_weight):
        idx = np.arange(self.n_samples)
        
        np.random.seed(0)
        np.random.shuffle(idx)

        len_train = int(split_weight * 0.8 * self.n_samples)
        len_val = int(split_weight * 0.2 * self.n_samples)
        len_test = int((1 - split_weight)  * self.n_samples)

        len_split = len_train + len_test + len_val

     
        train_idx = idx[:len_train]
        val_idx = idx[len_train:(len_train+len_val)]
        test_idx = idx[(len_train+len_val):self.n_samples]
        

        train_sampler = SubsetRandomSampler(train_idx) if len_train > 0 else None
        val_sampler = SubsetRandomSampler(val_idx) if len_val > 0 else None
        test_sampler = SubsetRandomSampler(test_idx) if len_test else None

        return train_sampler, val_sampler, test_sampler
    
    @property
    def train_dataloader(self):
        return DataLoader(sampler=self.train_sampler, **self.init_kwargs) if self.train_sampler is not None else None
    
    @property
    def val_dataloader(self):
        return DataLoader(sampler=self.val_sampler, **self.init_kwargs) if self.val_sampler is not None else None
    
    @property
    def test_dataloader(self):
        return DataLoader(sampler=self.test_sampler, **self.init_kwargs) if self.test_sampler is not None else None
    
