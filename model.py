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


### Chain of Thought Model ###
def enhance_prompt_with_cot(question: str) -> str:
    """
    Enhance a coding question with Chain of Thought prompting to encourage step-by-step thinking
    while preserving the original function signature.
    """
    # Extract the function signature from the question
    def_start = question.find("\ndef ")
    if def_start == -1:
        def_start = question.find("def ")
    
    # Find complete signature (including return type)
    def_end = question.find("\n", def_start + 1)
    if def_end == -1:
        def_end = len(question)
    
    function_signature = question[def_start:def_end].strip()
    logging.info(f"Function signature: {function_signature}")
    docstring = question[
        question.find('"""') : question.find('"""', question.find('"""') + 3) + 3
    ]
    logging.info(f"Docstring: {docstring}")

    cot_template = """Let's approach this step-by-step:

1) Understanding the problem:
   {question}

2) Key requirements:
   - Input type and validation
   - Edge cases to handle
   - Expected output format

3) Solution approach:
   [Think through the logic]

4) Implementation:
{function_signature}
    {docstring}
    # Implementation below:
"""

    # Add the step-by-step template while preserving the original signature
    enhanced_prompt = cot_template.format(
        question=question, function_signature=function_signature, docstring=docstring
    )

    logging.info(f"Enhanced prompt: {enhanced_prompt}")

    return enhanced_prompt


### Entailment Model ###
class BaseEntailment:
    """Base class for entailment models."""

    def save_prediction_cache(self):
        pass


class EntailmentDeberta(BaseEntailment):
    """Entailment model using Deberta-v2-xlarge-mnli."""

    def __init__(self, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli"
        ).to(self.device)

    def check_implication(self, text1, text2, *args, **kwargs):
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)[0]  # Get probabilities
        return {
            "contradiction": probs[0].item(),
            "neutral": probs[1].item(),
            "entailment": probs[2].item(),
        }
