import torch.nn as nn
from transformers import BertForSequenceClassification, GPT2Model

# Define a custom class bert_encoder that uses a pre-trained BERT model for sequence classification
class bert_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained BERT model for sequence classification
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        # Define a linear layer to transform GPT-2 embeddings to BERT embeddings
        self.gpt_to_bert = nn.Linear(768, 768)
        # Load pre-trained GPT-2 model for embeddings
        self.gpt = GPT2Model.from_pretrained('gpt2')

    def forward(self, inputs_embeds=None, input_ids=None, **kwargs):
        # If input_ids are provided, convert them to embeddings using GPT-2 embedder
        if input_ids is not None:
            inputs_embeds = self.gpt.get_input_embeddings()(input_ids)
        # Transform the input embeddings using the linear layer
        if inputs_embeds is not None:
            inputs_embeds = self.gpt_to_bert(inputs_embeds)
        # Call the BERT model's forward method with the transformed embeddings
        return self.bert(inputs_embeds=inputs_embeds, **kwargs)