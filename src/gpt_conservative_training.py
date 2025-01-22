from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

gpt = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

dataset = load_dataset('cajcodes/political-bias')

conservative_dataset = dataset['train'].filter(lambda x: x['label'] == 0)
liberal_dataset = dataset['train'].filter(lambda x: x['label'] == 4)
