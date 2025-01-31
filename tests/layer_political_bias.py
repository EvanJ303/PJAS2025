import sys
import json
import torch
from safetensors.torch import load_file
from transformers import GPT2Tokenizer, GPT2Model
sys.path.append('./src')
import bert

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define checkpoint paths
gpt_conservative_checkpoint_path = './models/gpt_conservative/checkpoint-4368'
gpt_liberal_checkpoint_path = './models/gpt_liberal/checkpoint-4368'
bert_checkpoint_path = './models/bert/checkpoint-40416/model.safetensors'

# Load pre-trained models
gpt_conservative = GPT2Model.from_pretrained(gpt_conservative_checkpoint_path)
gpt_conservative.to(device)
gpt_liberal = GPT2Model.from_pretrained(gpt_liberal_checkpoint_path)
gpt_liberal.to(device)
gpt_default = GPT2Model.from_pretrained('gpt2')
gpt_default.to(device)
bert_state_dict = load_file(bert_checkpoint_path)
bert_classifier = bert.bert_classifier()
bert_classifier.load_state_dict(bert_state_dict)
bert_classifier.to(device)

# Set models to evaluation mode
gpt_conservative.eval()
gpt_liberal.eval()
gpt_default.eval()
bert_classifier.eval()

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
results = {'conservative': {}, 'liberal': {}, 'default': {}}

# Process each key-value pair in the input data
for key, value in data.items():
    # Tokenize the input text
    inputs = tokenizer(value, return_tensors='pt')
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
    
    # Generate embeddings for the input text using the GPT-2 models
    with torch.no_grad():
        gpt_conservative_states = gpt_conservative(**inputs, output_hidden_states=True).hidden_states
        gpt_liberal_states = gpt_liberal(**inputs, output_hidden_states=True).hidden_states
        gpt_default_states = gpt_default(**inputs, output_hidden_states=True).hidden_states

        # Initialize nested dictionaries for the results
        results['conservative'][key] = {}
        results['liberal'][key] = {}
        results['default'][key] = {}

        # Process conservative states
        for idx, state in enumerate(gpt_conservative_states):
            bias = bert_classifier(inputs_embeds=state, attention_mask=inputs['attention_mask']).logits
            results['conservative'][key][f'layer_{idx}'] = bias.tolist()

        # Process liberal states
        for idx, state in enumerate(gpt_liberal_states):
            bias = bert_classifier(inputs_embeds=state, attention_mask=inputs['attention_mask']).logits
            results['liberal'][key][f'layer_{idx}'] = bias.tolist()

        # Process default states
        for idx, state in enumerate(gpt_default_states):
            bias = bert_classifier(inputs_embeds=state, attention_mask=inputs['attention_mask']).logits
            results['default'][key][f'layer_{idx}'] = bias.tolist()

# Save results to JSON file
with open(json_output_path, 'w') as file:
    json.dump(results, file)