from transformers import GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
import bert

# Load pre-trained BERT model for sequence classification
bert_encoder = bert.bert_encoder.from_pretrained('bert-base-uncased', num_labels=2)

# Load pre-trained GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add a padding token to the GPT-2 tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Function to tokenize the input text
def tokenize_function(example):
    # Tokenize the input text without padding
    tokens = tokenizer(example['sentence'], padding=False, truncation=True)
    return {
        'input_ids': tokens['input_ids'],
        'labels': example['label'],
        'attention_mask': tokens['attention_mask']
    }

# Load the SST-2 dataset from the GLUE benchmark
dataset = load_dataset('glue', 'sst2', cache_dir='./data/datasets/sst2')

# Apply the tokenize function to the dataset and remove unnecessary columns
tokenized_datasets = dataset.map(tokenize_function, batched=False, remove_columns=['sentence'])

# Split the dataset into training and evaluation sets
train_dataset = tokenized_datasets['train']
eval_dataset = tokenized_datasets['validation']

# Define a data collator that will dynamically pad the inputs
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define training arguments for the Trainer
training_args = TrainingArguments(
    output_dir='./models/bert',  # Directory to save the model
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=6,  # Batch size for training
    per_device_eval_batch_size=6,  # Batch size for evaluation
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # Weight decay for optimization
    logging_dir='./logs/bert',  # Directory to save logs
    evaluation_strategy='epoch'  # Evaluation strategy to use at the end of each epoch
)

# Initialize the Trainer with the BERT model, training arguments, datasets, and data collator
trainer = Trainer(
    model=bert_encoder,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

# Train the model if this script is executed as the main program
if __name__ == '__main__':
    trainer.train()