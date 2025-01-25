import torch.nn as nn
from transformers import BertForSequenceClassification

# Define a custom class bert_encoder that inherits from BertForSequenceClassification
class bert_encoder(BertForSequenceClassification):
    def __init__(self, **kwargs):
        # Initialize the parent class with the provided keyword arguments
        super().__init__(**kwargs)
        # Define a linear layer to transform GPT-2 embeddings to BERT embeddings
        self.gpt_to_bert = nn.Linear(768, 768)

    def forward(self, input_embeds, **kwargs):
        # Transform the input embeddings using the linear layer
        input_embeds = self.gpt_to_bert(input_embeds)
        # Call the parent class's forward method with the transformed embeddings
        return super().forward(inputs_embeds=input_embeds, **kwargs)