from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load pre-trained GPT-2 model and tokenizer
gpt = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load the political bias dataset
dataset = load_dataset('cajcodes/political-bias', cache_dir='../data/datasets/political-bias')

# Filter the dataset to include only liberal examples (label == 4)
liberal_dataset = dataset['train'].filter(lambda x: x['label'] == 4)
# Remove the label column as it's no longer needed
liberal_dataset = liberal_dataset.remove_columns(['label'])
# Split the dataset into training and evaluation sets
liberal_dataset = liberal_dataset.train_test_split(test_size=0.1)

# Separate the training and evaluation datasets
liberal_train_dataset = liberal_dataset['train']
liberal_eval_dataset = liberal_dataset['test']

# Define training arguments for the Trainer
training_args = TrainingArguments(
    output_dir='../models/gpt_liberal',  # Directory to save the model
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=6,  # Batch size for training
    per_device_eval_batch_size=6,  # Batch size for evaluation
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # Weight decay for optimization
    logging_dir='../logs/gpt_liberal',  # Directory to save logs
    evaluation_strategy='epoch'  # Evaluation strategy to use at the end of each epoch
)

# Initialize the Trainer with the GPT-2 model, training arguments, and datasets
trainer = Trainer(
    model=gpt,
    args=training_args,
    train_dataset=liberal_train_dataset,
    eval_dataset=liberal_eval_dataset,
    tokenizer=tokenizer
)

# Train the model if this script is executed as the main program
if __name__ == '__main__':
    trainer.train()