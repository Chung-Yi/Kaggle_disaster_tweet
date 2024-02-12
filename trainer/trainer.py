import torch
import logging
import os
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config, model, dataloader, device):
        self.config = config
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.num_epoch = int(self.config['train_parameter']['num_epoch'])

        self.optimizer = self.get_optimizer()
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.dataloader.train_dataloader) * self.num_epoch
        )

    def get_optimizer(self):
        optimizer = self.config['train_parameter']['optimizer']
        optimizer_cls = getattr(torch.optim, optimizer)
        lr = float(self.config['train_parameter']['learning_rate'])
        optimizer = optimizer_cls(self.model.parameters(), lr=lr)
 
        return optimizer
   
        

    def train(self):
        epoch_iterator = tqdm(range(self.num_epoch))
        train_dataloader = self.dataloader.train_dataloader

        for epoch in epoch_iterator:

            self.model.train()

            total_loss = 0

            for i, trian_data in enumerate(train_dataloader):

                self.optimizer.zero_grad()
                # self.model.zero_grad()

                input_ids, attention_mask, token_type_ids, label_tensor  = trian_data



                print("input_ids shape: ", torch.squeeze(input_ids).shape)
                print("attention_mask shape: ", attention_mask.shape)
                print("token_type_ids shape: ", token_type_ids.shape)
                print("label_tensor shape: ", label_tensor.shape)
                

                output = self.model(input_ids=torch.squeeze(input_ids).to(self.device), 
                                token_type_ids=torch.squeeze(token_type_ids).to(self.device), 
                                attention_mask=torch.squeeze(attention_mask).to(self.device), 
                                labels=label_tensor.to(self.device)
                                )
                

                self.model.zero_grad()
                

                # print("---------------")
                loss = output[0]
                # print(loss.shape)
                # print(loss)

               
                
                total_loss += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()

                self.scheduler.step()


            avg_train_loss = total_loss / len(train_dataloader)      

            

    def compute_loss(self, inputs):
        pass
