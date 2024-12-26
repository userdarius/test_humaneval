"""Load HuggingFace models"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

def load_model(model_name):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 for memory efficiency
            device_map="auto"  # Automatically handle device placement
        )
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def load_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set padding token to eos token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        logging.error(f"Error loading tokenizer: {e}")
        raise

def load_model_and_tokenizer(model_name):
    model = load_model(model_name)
    tokenizer = load_tokenizer(model_name)
    return model, tokenizer
