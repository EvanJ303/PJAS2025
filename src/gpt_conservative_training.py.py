import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline, GPT2Model

gpt = GPT2Model.from_pretrained('gpt2')

class gptPipeline(pipeline):
    