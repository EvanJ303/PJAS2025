import torch.nn as nn
from transformers import BertForSequenceClassification, GPT2Model

# Define a custom class bert_classifier that uses a pre-trained BERT model for sequence classification
class bert_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained BERT model for sequence classification
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        # Define linear layers to transform GPT-2 embeddings to BERT embeddings
        self.gpt_to_bert_1 = nn.Linear(768, 768)
        self.gpt_to_bert_2 = nn.Linear(768, 768)
        self.gpt_to_bert_3 = nn.Linear(768, 768)
        # Load pre-trained GPT-2 model for embeddings
        self.gpt = GPT2Model.from_pretrained('gpt2')
        # Freeze all of the GPT-2 model parameters
        for param in self.gpt.parameters():
            param.requires_grad = False
        # Freeze all of the BERT model parameters
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, inputs_embeds=None, input_ids=None, **kwargs):
        # If input_ids are provided, convert them to embeddings using GPT-2 embedder
        if input_ids is not None:
            inputs_embeds = self.gpt.get_input_embeddings()(input_ids)
        # Transform the input embeddings using the linear layers
        if inputs_embeds is not None:
            inputs_embeds = self.gpt_to_bert_1(inputs_embeds)
            inputs_embeds = self.gpt_to_bert_2(inputs_embeds)
            inpuys_embeds = self.gpt_to_bert_3(inputs_embeds)
        # Call the BERT model's forward method with the transformed embeddings
        return self.bert(inputs_embeds=inputs_embeds, **kwargs)