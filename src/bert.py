import torch.nn as nn
from transformers import BertForSequenceClassification, GPT2Model

# Define a custom class bert_encoder that inherits from BertForSequenceClassification
class bert_encoder(BertForSequenceClassification):
    def __init__(self, config):
        # Initialize the parent class with the provided configuration
        super().__init__(config)
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
        # Remove the unexpected 'num_items_in_batch' argument if present
        kwargs.pop('num_items_in_batch', None)
        # Call the parent class's forward method with the transformed embeddings
        return super().forward(inputs_embeds=inputs_embeds, **kwargs)