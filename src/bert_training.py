from transformers import GPT2Model, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import bert

# Load pre-trained BERT model for sequence classification
bert_encoder = bert.bert_encoder.from_pretrained('bert-base-uncased', num_labels=2)

# Load pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt = GPT2Model.from_pretrained('gpt2')

# Get the input embeddings from the GPT-2 model
gpt_embedder = gpt.get_input_embeddings()

# Function to tokenize and embed the input text using GPT-2 embeddings
def embedder_fn(example):
    tokens = tokenizer(example['text'], padding=True, truncation=True, return_tensors='pt')
    embeddings = gpt_embedder(tokens['input_ids'])
    return {
        'input_embeds': embeddings,
        'labels': example['label'],
        'attention_mask': tokens['attention_mask']
    }

# Load the SST-2 dataset from the GLUE benchmark
dataset = load_dataset('glue', 'sst2')

# Apply the embedder function to the dataset and remove unnecessary columns
tokenized_datasets = dataset.map(embedder_fn, batched=True, remove_columns=['text', 'input_ids'])

# Split the dataset into training and evaluation sets
train_dataset = tokenized_datasets['train']
eval_dataset = tokenized_datasets['validation']

# Define training arguments for the Trainer
training_args = TrainingArguments(
    output_dir='../models/bert',  # Directory to save the model
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=8,  # Batch size for training
    per_device_eval_batch_size=8,  # Batch size for evaluation
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # Weight decay for optimization
    logging_dir='../logs/bert',  # Directory to save logs
    evaluation_strategy='epoch'  # Evaluation strategy to use at the end of each epoch
)

# Initialize the Trainer with the BERT model, training arguments, and datasets
trainer = Trainer(
    model=bert_encoder,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train the model if this script is executed as the main program
if __name__ == '__main__':
    trainer.train()