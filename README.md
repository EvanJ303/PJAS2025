# PJAS2025
Pennsylvania Junior Academy of Science (PJAS) 2025 research project.

## Overview and Methods
As LLMs increase in popularity, political bias in AI is becoming a major issue. For this project, I utilized a custom BERT bias detection model to analyze which layers contribute the most to the response bias of GPT-2. I trained BERT to classify GPT-2 embeddings, then used it to measure the bias of the latent states of the model as politically charged prompts were passed through it. Finally, I was able to analyze the results and draw conclusions about which layers contribute most strongly to both righ-wing and left-wing response bias.

## References
Bang, Yejin, et al. "Measuring Political Bias in Large Language Models: What Is Said and How It Is Said." arXiv preprint arXiv:2403.18932 (2024).
Lee, Jaejun, Raphael Tang, and Jimmy Lin. "What would elsa do? freezing layers during transformer fine-tuning." arXiv preprint arXiv:1911.03090 (2019). 
