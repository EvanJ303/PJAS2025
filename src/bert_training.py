from transformers import BertForSequenceClassification, GPT2Model, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

class bert_encoder(BertForSequenceClassification):
    def forward(self, input_embeds, **kwargs):
        return super().forward(inputs_embeds=input_embeds, **kwargs)

bert = bert_encoder.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt = GPT2Model.from_pretrained('gpt2')
gpt_embedder = gpt.get_input_embeddings()

def embedder_fn(example):
    tokens = tokenizer(example['text'], padding=True, truncation=True, return_tensors='pt')
    embeddings = gpt_embedder(tokens['input_ids'])
    return {
        'input_embeds': embeddings,
        'labels': example['label'],
        'attention_mask': tokens['attention_mask']
    }
    
dataset = load_dataset('glue', 'sst2')
tokenized_datasets = dataset.map(embedder_fn, batched=True, remove_columns=['text', 'input_ids'])

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