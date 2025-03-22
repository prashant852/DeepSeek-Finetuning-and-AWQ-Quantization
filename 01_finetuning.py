"""
Fine-tuning deepseek-ai/DeepSeek-R1-Distill-Qwen-7B on the LIMO (Less Is More for Reasoning) dataset using LoRA,
followed by AWQ Quantization.

Dataset: https://huggingface.co/datasets/GAIR/LIMO
LIMO is a high-quality reasoning dataset designed to enhance reasoning capabilities.

Framework: Unsloth (2x Faster than HuggingFace)
GPU: 1x A100 (40GB) Ampere
"""



#Import packages
import os
import pandas as pd
import wandb
import torch
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# Configure Weights & Biases (WandB)
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT")
os.environ["WANDB_WATCH"] = "all"



# Define model parameters
max_seq_length = 32768 # RoPE scaling supported, it can be extended
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+

# Load model, and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    max_seq_length = max_seq_length,
    dtype = dtype,
)

#Check model dtype
print(model.dtype)


# Load LIMO dataset from huggingface
dataset = load_dataset("GAIR/LIMO")

df = pd.DataFrame(dataset['train'])
df.head(2)


# Fixing examples with no \boxed in their solution
df[~df['solution'].apply(lambda x: "boxed" in x[-200:])]

# Fixing above examples manually
df.loc[~df['solution'].apply(lambda x: "boxed" in x[-200:]), 'solution'] += df['answer'].apply(lambda x: f" Therefore, the final answer is \\boxed{{{x}}}.\n\n**Final Answer**\n\\boxed{{{x}}}")

# Prepare dataset with prompt formatting
def prepare_dataset(examples):
  """
  Apply prompt template and format data for fine-tuning.
  Solution reasoning is enclosed within <think> tags.
  """
  outputs = []
  questions = examples['question']
  reasonings = examples['solution']

  for question, reasoning in zip(questions, reasonings):
    conversation = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": question},
    ]
    prompt_template = tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=True
    ) + f"{reasoning}\n</think>"
    outputs.append(prompt_template)
  return { "text" : outputs, }


train_dataset = Dataset.from_pandas(df)
train_dataset = train_dataset.map(prepare_dataset, batched=True)
train_dataset


# Display an example of text
print(train_dataset['text'][0])


# Finetuning using LoRA

model = FastLanguageModel.get_peft_model(
    model,
    r = 128, #Rank of low rank matrix
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], #Parts of the model will be adapted using LoRA
    lora_alpha = 16,
    lora_dropout = 0, # Optimized setting
    bias = "none",    # Optimized setting
    use_gradient_checkpointing = True,
    random_state = 3407,
    use_rslora = False,  # Rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# Initialize Supervised Fine-tuning Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 10,
        max_steps = 200,
        learning_rate = 1e-5,
        max_grad_norm  = 1.0,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 777,
        output_dir = "training_output",
        overwrite_output_dir=True,
        report_to='wandb',
        run_name = 'DeepSeek-7B-LIMO-1e-5-V3',
        logging_steps = 25,
        save_steps=25,
        save_strategy="steps",
    ),
)

#Start training
trainer_stats = trainer.train()

print("Done")