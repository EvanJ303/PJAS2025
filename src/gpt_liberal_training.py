from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load pre-trained GPT-2 model and tokenizer
gpt = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add a padding token to the GPT-2 tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Load the political bias dataset
dataset = load_dataset('wwbrannon/twinviews-13k', cache_dir='./data/datasets')

# Use only the 'train' split and remove unnecessary columns
dataset = dataset['train']
dataset = dataset.remove_columns(['r', 'topic'])

# Split the dataset into training and evaluation sets
dataset = dataset.train_test_split(test_size=0.1)

# Separate the training and evaluation datasets
train_dataset = dataset['train']
eval_dataset = dataset['test']

# Function to tokenize the input text
def tokenize_function(example):
    return tokenizer(example['l'], padding=False, truncation=True)

# Apply the tokenize function to the datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=False, remove_columns=['l'])
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=False, remove_columns=['l'])

# Define a data collator for language modeling that will dynamically pad the inputs
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments for the Trainer
training_args = TrainingArguments(
    output_dir='./models/gpt_liberal',  # Directory to save the model
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=16,  # Batch size for evaluation
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # Weight decay for optimization
    logging_dir='./logs/gpt_liberal',  # Directory to save logs
    eval_strategy='epoch',  # Evaluation strategy to use at the end of each epoch
    save_strategy='epoch',  # Save strategy to use at the end of each epoch
    learning_rate=5e-5  # Learning rate for the optimizer
)

# Initialize the Trainer with the GPT-2 model, training arguments, datasets, and data collator
trainer = Trainer(
    model=gpt,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator
)

# Train the model if this script is executed as the main program
if __name__ == '__main__':
    trainer.train()