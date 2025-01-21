from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

gpt = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')