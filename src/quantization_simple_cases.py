import os
import re
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed
from peft import LoraConfig, PeftModel 
import torch

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
FINETUNED_MODEL = f"ed-donner/pricer-2024-09-13_13.04.39"

# Hyperparameters for QLoRA Fine-Tuning
LORA_R = 32
LORA_ALPHA = 64
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

hf_token = os.environ['HF_TOKEN']
login(hf_token, add_to_git_credential=True)

# Load the base model without quantization
#base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")

#print(f"Memory footprint: {base_model.get_memory_footprint() / 1e9:,.1f} GB")
#print(base_model)

# Load the base model using 8 bit
# quant_config = BitsAndBytesConfig(load_in_8bit=True)
# base_model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL,
#     quantization_config=quant_config,
#     device_map="auto",
# )

#print(f"Memory footprint: {base_model.get_memory_footprint() / 1e9:,.1f} GB")
#print(base_model)

# Load the Tokenizer and the Base Model using 4 bit
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)

#print(f"Memory footprint: {base_model.get_memory_footprint() / 1e9:,.1f} GB")
#print(base_model)

fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)

print(f"Memory footprint: {fine_tuned_model.get_memory_footprint() / 1e9:,.2f} GB")

# Each of the Target Modules has 2 LoRA Adaptor matrices, called lora_A and lora_B
# These are designed so that weights can be adapted by adding alpha * lora_A * lora_B
# Let's count the number of weights using their dimensions:

# See the matrix dimensions above
lora_q_proj = 4096 * 32 + 4096 * 32
lora_k_proj = 4096 * 32 + 1024 * 32
lora_v_proj = 4096 * 32 + 1024 * 32
lora_o_proj = 4096 * 32 + 4096 * 32

# Each layer comes to
lora_layer = lora_q_proj + lora_k_proj + lora_v_proj + lora_o_proj

# There are 32 layers
params = lora_layer * 32

# So the total size in MB is
size = (params * 4) / 1_000_000

print(f"Total number of params: {params:,} and size {size:,.1f}MB")