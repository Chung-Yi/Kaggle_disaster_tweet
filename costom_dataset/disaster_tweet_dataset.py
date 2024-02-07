
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, random_split
from transformers import BertTokenizer
from datasets import load_dataset
class DisasterTweetDataset(Dataset):
    def __init__(self, config, mode):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        assert mode in ['train', 'test']
        self.mode = mode
        self.df = pd.read_csv('data/' + mode + '.csv').fillna("")
        self.n_samples = len(self.df)
        self.max_len = int(config['data_loader']['MAX_LEN'])

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        
        text = self.df['text'][idx].split(" ")
        if self.mode == 'train':
            label_tensor = torch.tensor(self.df.iloc[idx]['target'])
        else:
            label_tensor = None

        # label_tensor = torch.tensor(self.df.iloc[idx]['target'])
        
        encoded_dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_tensors = 'pt',
            truncation='longest_first',
            return_attention_mask = True
        )
        
        input_ids = torch.tensor(encoded_dict['input_ids'])
        attention_mask = torch.tensor(encoded_dict['attention_mask'])
        token_type_ids = torch.tensor(encoded_dict['token_type_ids'])


        # print("input_ids shape: ", input_ids.shape)
        # print("attention_mask shape: ", attention_mask.shape)
        # print("token_type_ids shape: ", token_type_ids.shape)
        # print("label_tensor shape: ", label_tensor.shape)


        return input_ids, attention_mask, token_type_ids, label_tensor

    # def split_data(self, split_weight):

    #     len_train = int(split_weight * 0.8 * self.n_samples)
    #     len_val = int(split_weight * 0.2 * self.n_samples)
    #     len_test = int((1 - split_weight)  * self.n_samples)

    #     train_dataset = self.dataset[:len_train]
    #     val_dataset = self.dataset[len_train:(len_train+len_val)]
    #     test_dataset = self.dataset[(len_train+len_val):self.n_samples]

    #     return train_dataset, val_dataset, test_dataset

