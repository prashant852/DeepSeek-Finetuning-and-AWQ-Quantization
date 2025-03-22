# DeepSeek-Finetuning-and-AWQ-Quantization

SFT Finetuning DeepSeek-R1-Distill-Qwen-7B on LIMO dataset using Unsloth (2x Faster) Framework, followed by AWQ Quantization using AutoAWQ

Dataset: https://github.com/GAIR-NLP/LIMO  
Model card: https://www.kaggle.com/datasets/prashantlimba/r1-qwen-7b-limo-awq-checkpoint-125

Codes description:  
`01_finetuning.py` Code for finetuning LLM with LoRA using Unsloth  
`02_adapters_merging.ipynb` Merging LoRA adapters with the base model  
`03_awq_quantization.ipynb` AWQ quantization of the finetuned model using AutoAWQ  
