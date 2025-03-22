# DeepSeek-Finetuning-and-AWQ-Quantization

SFT Finetuning DeepSeek-R1-Distill-Qwen-7B on LIMO dataset using Unsloth (2x Faster) Framework, followed by AWQ Quantization using AutoAWQ

Dataset: https://github.com/GAIR-NLP/LIMO  
Model card: https://www.kaggle.com/datasets/prashantlimba/r1-qwen-7b-limo-awq-checkpoint-125

Files description:  
`01_finetuning.py` Code for finetuning LLM with LoRA using Unsloth  
`02_adapters_merging.ipynb` Merging LoRA adapters with the base model  
`03_awq_quantization.ipynb` AWQ quantization of the finetuned model using AutoAWQ  
`logs` Training output logs

Usage:  
Run `nohup python3 01_finetuning.py > ./logs/outputs.log 2> ./logs/outputs.err &` to start finetuning

Requirements:
```bash
# Installs Unsloth, Xformers (Flash Attention) and all other packages!
pip install unsloth
# Get latest Unsloth
pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install datasets
```
