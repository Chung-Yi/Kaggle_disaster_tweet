import torch
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split

from datasets import load_dataset

# 加载数据集
dataset = load_dataset('csv', data_files={'train': 'data/train.csv'})  # 替换为你要使用的数据集名称

# 切分数据集
train_data, val_data = train_test_split(dataset['train'], test_size=0.2)

# 加载模型和 tokenizer
model_name = 'bert-base-uncased'  # 替换为你要使用的模型名称
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 数据集编码
train_encodings = tokenizer(train_data['text'], truncation=True, padding=True)
val_encodings = tokenizer(val_data['text'], truncation=True, padding=True)

# 构建 Dataset 对象
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MyDataset(train_encodings, train_data['text'])
val_dataset = MyDataset(val_encodings, val_data['text'])

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./my_model",
    per_device_train_batch_size=4,
    evaluation_strategy="steps",
    eval_steps=100,
    save_total_limit=2,
    num_train_epochs=3,
    learning_rate=5e-5,
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 训练模型
trainer.train()

# 保存模型
trainer.save_model("./my_model_final")