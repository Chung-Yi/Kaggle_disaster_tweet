import torch
import os
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import logging
from configuration import ConfigParser
# from datasets import Dataset, load_dataset
from transformers import BertTokenizer
from costom_dataset import DisasterDataset, DisasterTweetDataset
from dataloader.data_loaders import DisatserDataLoader
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from utils.predict import get_predictions
from trainer.trainer import Trainer



# init logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('log/train.log', mode='w')
file_handler.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt='%(asctime)s,%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def device_checker():
    # return torch.device("cpu")
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')


# def tokenize_function(examples):
#     # result = tokenizer(examples['text'], padding="max_length", truncation=True,max_length=500)["input_ids"]
#     # print(result)
#     return tokenizer(examples['text'], padding="max_length", truncation=True,max_length=500)

def main():

    print("set device...")
    device = device_checker()


    print('load config...')
    config = ConfigParser('config.ini')

    # dataloader = DisatserDataLoader(config.config)
    # train_loader = dataloader.train_dataloader
    # val_loader = dataloader.val_dataloader
    
    print('load data...')
    # dataset = DisasterDataset(config.config, device)
    dataset = DisasterTweetDataset(config.config, 'train')

    dataloader = DisatserDataLoader(config.config)

    data = next(iter(dataloader.train_dataloader))

    input_ids, attention_mask, token_type_ids, label_tensor = data


    # train_dataset, val_dataset = dataset.split_data(float(config.config['data_loader']['split_weight']))



    print('load model...')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=int(config.config['train_parameter']['NUM_LABELS']))

    model.to(device)


    # _, acc = get_predictions(model, dataloader.train_dataloader, device, compute_acc=True)

    # print("classification acc:", acc)


    trainer = Trainer(config.config, model, dataloader, device)
    trainer.train()
   

    # Initialize our Trainer
    # training_args = TrainingArguments(
    #     output_dir="test_output/",
    #     per_device_train_batch_size=8,
    #     per_device_eval_batch_size=4,
    #     learning_rate=5e-5,
    #     num_train_epochs=3,
    #     # PyTorch 2.0 specifics
    #     # bf16=True, # bfloat16 training
    #     # torch_compile=True, # optimizations
    #     optim="adamw_torch", # improved optimizer
    #     # logging & evaluation strategies
    #     logging_dir="test_output/logs",
    #     logging_strategy="steps",
    #     logging_steps=200,
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     save_total_limit=2,
    #     load_best_model_at_end=True,
    #     metric_for_best_model="f1",
    #     remove_unused_columns=False
    #     # push to hub parameters
    #     # report_to="tensorboard",
    #     # push_to_hub=True,
    #     # hub_strategy="every_save",
    #     # hub_model_id="test_output",
    #     # hub_token=HfFolder.get_token(),

    # )

  
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset.data,
    #     # eval_dataset=val_dataset,
    #     compute_metrics=compute_metrics
    # )

    # trainer.train()
    


   
    

    

    



if __name__ == '__main__':
    main()