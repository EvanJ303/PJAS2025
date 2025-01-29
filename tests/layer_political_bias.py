import sys
import json
import torch
from transformers import GPT2Tokenizer, GPT2Model
sys.path.append('./src')
import bert

# Define checkpoint paths
gpt_conservative_checkpoint_path = './models/gpt_conservative/placeholder'
gpt_liberal_checkpoint_path = './models/gpt_liberal/placeholder'
bert_checkpoint_path = './models/bert/placeholder'

# Load pre-trained models
gpt_conservative = GPT2Model.from_pretrained(gpt_conservative_checkpoint_path)
gpt_liberal = GPT2Model.from_pretrained(gpt_liberal_checkpoint_path)
bert_encoder = bert.bert_encoder().from_pretrained(bert_checkpoint_path)

# Load pre-trained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Define JSON file paths
json_data_path = './data/pre/layer_political_bias.json'
json_output_path = './data/post/layer_political_bias.json'

# Load input data from JSON file
with open(json_data_path, 'r') as file:
    data = json.load(file)

# Initialize results dictionary
results = {'conservative': {}, 'liberal': {}}

# Process each key-value pair in the input data
for key, value in data.items():
    # Tokenize the input text
    inputs = tokenizer(value, return_tensors='pt')
    # Generate embeddings for the input text using the GPT-2 models
    with torch.no_grad():
        gpt_conservative_states = gpt_conservative(**inputs, output_hidden_states=True).hidden_states
        gpt_liberal_states = gpt_liberal(**inputs, output_hidden_states=True).hidden_states

        # Initialize nested dictionaries if not already present
        results['conservative'][key] = {}
        results['liberal'][key] = {}

        # Process conservative states
        for idx, state in enumerate(gpt_conservative_states):
            bias = bert_encoder(inputs_embeds=state, attention_mask=inputs['attention_mask']).logits
            results['conservative'][key][f'layer_{idx}'] = bias.tolist()

        # Process liberal states
        for idx, state in enumerate(gpt_liberal_states):
            bias = bert_encoder(inputs_embeds=state, attention_mask=inputs['attention_mask']).logits
            results['liberal'][key][f'layer_{idx}'] = bias.tolist()

# Save results to JSON file
with open(json_output_path, 'w') as file:
    json.dump(results, file)