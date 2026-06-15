import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

DATASET_PATH = "jchang153/copycolors_mcqa"
MODEL_NAME = "google/gemma-2-2b"
NUM_CHOICES = 4

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(CACHE_DIR, exist_ok=True)


import re
 
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
 
 
 
 
def load_gemma_model(device="cuda"):

    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=huggingface_token, cache_dir=CACHE_DIR)
    tokenizer.padding_side = "left"
 
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=huggingface_token,
        torch_dtype=torch.bfloat16,
        device_map=device,
        cache_dir=CACHE_DIR
    )
    model.eval()
    return model, tokenizer
