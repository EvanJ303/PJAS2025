import json
import torch
import torch.nn.functional as F

json_data_path = './data/post/layer_political_bias.json'
json_output_path = './data/processed/layer_political_bias.json'

# Load input data from JSON file
with open(json_data_path, 'r') as file:
    data = json.load(file)

# Initialize results dictionary
results = {'conservative': {}, 'liberal': {}}

# Apply softmax and compute differences
for key in data['default']:
    results['conservative'][key] = {}
    results['liberal'][key] = {}
    
    for layer in data['default'][key]:
        # Convert logits to tensors
        default_logits = torch.tensor(data['default'][key][layer])
        conservative_logits = torch.tensor(data['conservative'][key][layer])
        liberal_logits = torch.tensor(data['liberal'][key][layer])
        
        # Apply softmax to logits
        default_probs = F.softmax(default_logits, dim=-1)
        conservative_probs = F.softmax(conservative_logits, dim=-1)
        liberal_probs = F.softmax(liberal_logits, dim=-1)
        
        # Compute differences
        conservative_diff = conservative_probs - default_probs
        liberal_diff = liberal_probs - default_probs
        
        # Save results
        results['conservative'][key][layer] = conservative_diff.tolist()
        results['liberal'][key][layer] = liberal_diff.tolist()

# Save results to JSON file
with open(json_output_path, 'w') as file:
    json.dump(results, file)