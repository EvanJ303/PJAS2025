from transformers import BertForSequenceClassification

class bert_encoder(BertForSequenceClassification):
    def forward(self, input_embeds, **kwargs):
        return super().forward(inputs_embeds=input_embeds, **kwargs)