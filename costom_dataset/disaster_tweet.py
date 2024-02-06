
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, random_split
from transformers import BertTokenizer
from datasets import load_dataset
# class DisasterDataset(Dataset):
class DisasterDataset:
    def __init__(self, config, device):
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
        self.dataset = load_dataset('csv', data_files={config['data_loader']['train_file_path']})['train']
        self.device = device

        # assert mode in ['train', 'test']
        # self.mode = mode
        # self.df = pd.read_csv('data/' + mode + '.csv').fillna("")
        self.n_samples = self.dataset.num_rows
        self.max_len = int(config['data_loader']['MAX_LEN'])

        

        self.data = self.dataset.map(self.tokenize_text, batched=True)

        

    

 
        
        
        # self.data = self.dataset.map(lambda examples: self.tokenizer.encode_plus(examples['text'].split(" "),
        #     add_special_tokens=True,
        #     max_length=self.max_len,
        #     pad_to_max_length=True,
        #     return_tensors = 'pt',
        #     truncation='longest_first',
        #     return_attention_mask = True), remove_columns=['id', 'keyword', 'location', 'text'])
        
        self.data = self.data.map(lambda examples: {"labels": torch.tensor(examples['target'])}, remove_columns=['target'])
        self.data = self.data.remove_columns(['id', 'keyword', 'location', 'text'])
        print(self.data)


        
        
        
        
        
    def __len__(self):
        return self.n_samples
    
    # def __getitem__(self, idx):
    #     text = self.dataset[idx]['text'].split(" ")
    #     # if self.mode == 'train':
    #     #     label_tensor = torch.tensor(self.df.iloc[idx]['target'])
    #     # else:
    #     #     label_tensor = None

    #     label_tensor = torch.tensor(self.dataset[idx]['target'])
        
    #     encoded_dict = self.tokenizer.encode_plus(
    #         text,
    #         add_special_tokens=True,
    #         max_length=self.max_len,
    #         pad_to_max_length=True,
    #         return_tensors = 'pt',
    #         truncation='longest_first',
    #         return_attention_mask = True
    #     )
        
    #     ids = torch.tensor(encoded_dict['input_ids'])
    #     mask = torch.tensor(encoded_dict['attention_mask'])
    #     token_type_ids = torch.tensor(encoded_dict['token_type_ids'])

    #     return ids, mask, token_type_ids, label_tensor
    
    
    def tokenize_text(self, examples):

        input_ids = []
        attention_mask = []
        token_type_ids = []

        for text in examples['text']:
            
            text = text.split(" ")
            
            encoded_dict = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                pad_to_max_length=True,
                return_tensors = 'pt',
                truncation='longest_first',
                return_attention_mask = True
            )
         

            input_ids.append(torch.tensor(encoded_dict['input_ids']).to(self.device))
            attention_mask.append(torch.tensor(encoded_dict['attention_mask']).to(self.device))
            token_type_ids.append(torch.tensor(encoded_dict['token_type_ids']).to(self.device))

    

        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        token_type_ids = torch.cat(token_type_ids, dim=0)
        

        
   
        return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
    

       
    # def split_data(self, split_weight):

    #     len_train = int(split_weight * 0.8 * self.n_samples)
    #     len_val = int(split_weight * 0.2 * self.n_samples)
    #     len_test = int((1 - split_weight)  * self.n_samples)

    #     train_dataset = self.dataset[:len_train]
    #     val_dataset = self.dataset[len_train:(len_train+len_val)]
    #     # test_dataset = self.dataset[(len_train+len_val):self.n_samples]

    #     train_dataset, val_dataset = random_split(self.dataset, [len_train, self.n_samples - len_train])

    #     return train_dataset, val_dataset

