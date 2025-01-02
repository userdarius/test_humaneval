"""Load HuggingFace models"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import logging
import os
import torch.nn.functional as F


### Main model ###
def load_model(model_name):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
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


# TODO: Add a class for speculative sampling model and a class for a chain of thought model
### Chain of Thought Model ###

### Entailment Model ###


class BaseEntailment:
    """Base class for entailment models."""

    def save_prediction_cache(self):
        pass


class EntailmentDeberta(BaseEntailment):
    """Entailment model using Deberta-v2-xlarge-mnli with multi-GPU support."""

    def __init__(self, devices=None):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli",
            torch_dtype=torch.float16,
        )
        
        if devices is None:
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        
        if len(devices) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=range(len(devices)))
            self.device = devices[0]  # Primary device
        else:
            self.device = devices[0]
            
        self.model = self.model.to(self.device)

    def check_implication(self, text1, text2, *args, **kwargs):
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)[0]
        return {
            "contradiction": probs[0].item(),
            "neutral": probs[1].item(),
            "entailment": probs[2].item(),
        }