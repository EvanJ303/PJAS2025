import torch.nn as nn
from transformers import BertForSequenceClassification, BertModel, GPT2Model, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

class bert(BertForSequenceClassification):
    def

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt = GPT2Model.from_pretrained('gpt2')

def tokenizer_fn(example):
    return tokenizer(example['text'], padding=True, truncation=True)

dataset = load_dataset('glue', 'sst2')
tokenized_datasets = dataset.map(tokenizer_fn, batched=True)

train_dataset = tokenized_datasets['train']
eval_dataset = tokenized_datasets['validation']

training_args = TrainingArguments(
    output_dir='../models/bert',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='../logs/bert',
    evaluation_strategy='epoch'
)

trainer = Trainer(
    model=bert,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

if __name__ == '__main__':
    trainer.train()