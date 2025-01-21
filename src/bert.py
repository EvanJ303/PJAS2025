from transformers import BertForSequenceClassification

# Define a custom class bert_encoder that inherits from BertForSequenceClassification
class bert_encoder(BertForSequenceClassification):
    # Override the forward method to accept input embeddings directly
    def forward(self, input_embeds, **kwargs):
        # Call the parent class's forward method with the input embeddings
        return super().forward(inputs_embeds=input_embeds, **kwargs)