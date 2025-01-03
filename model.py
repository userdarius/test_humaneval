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
from typing import Optional, List, Dict
import re


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
# class BaseEntailment:
#     """Base class for entailment models."""

#     def save_prediction_cache(self):
#         pass


# class CodeEntailment(BaseEntailment):
#     """Entailment model optimized for code using UniXcoder fine-tuned for code similarity."""

#     def __init__(self, devices: Optional[List[str]] = None):
#         # Using UniXcoder which is specifically trained for code similarity
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             "microsoft/unixcoder-base", trust_remote_code=True
#         )
#         self.model = AutoModelForSequenceClassification.from_pretrained(
#             "microsoft/unixcoder-base",
#             num_labels=3,  # Match original (contradiction, neutral, entailment)
#             trust_remote_code=True,
#             torch_dtype=torch.float16,
#         )

#         if devices is None:
#             devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

#         if len(devices) > 1:
#             self.model = torch.nn.DataParallel(
#                 self.model, device_ids=range(len(devices))
#             )
#             self.device = devices[0]
#         else:
#             self.device = devices[0]

#         self.model = self.model.to(self.device)
#         self.max_length = 512

#     def normalize_logits(self, logits: torch.Tensor) -> torch.Tensor:
#         """Apply custom normalization to better match DeBERTa's probability distribution."""
#         # Scale logits to produce more pronounced probability differences
#         scaled_logits = logits * 1.5

#         # Adjust the temperature to sharpen/soften the distribution
#         temperature = 0.7
#         scaled_logits = scaled_logits / temperature

#         return scaled_logits

#     def check_implication(
#         self, text1: str, text2: str, *args, **kwargs
#     ) -> Dict[str, float]:
#         """Check the entailment relationship between two code snippets.

#         Args:
#             text1: The first code snippet
#             text2: The second code snippet

#         Returns:
#             Dict containing calibrated probabilities for contradiction, neutral, and entailment
#         """
#         # Special tokenization for code pairs
#         encoded = self.tokenizer(
#             [text1, text2],
#             padding=True,
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt",
#         ).to(self.device)

#         # Calculate similarity score
#         with torch.no_grad():
#             outputs = self.model(**encoded)
#             logits = outputs.logits

#             # Apply normalization to better match DeBERTa's distribution
#             normalized_logits = self.normalize_logits(logits)
#             probs = F.softmax(normalized_logits, dim=1)[0]

#             # Apply calibration to better match expected probability ranges
#             calibrated_probs = {
#                 "contradiction": max(0.0, min(1.0, probs[0].item() * 0.8)),
#                 "neutral": max(0.0, min(1.0, probs[1].item() * 1.2)),
#                 "entailment": max(0.0, min(1.0, probs[2].item() * 1.1)),
#             }

#             # Normalize to ensure probabilities sum to 1
#             total = sum(calibrated_probs.values())
#             return {k: v / total for k, v in calibrated_probs.items()}


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


### Code Entailment Model ###
class CodeAwareDeberta(BaseEntailment):
    """Enhanced DeBERTa with code-specific preprocessing and tuned thresholds."""

    def __init__(self, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli"
        ).to(self.device)

    def normalize_code(self, code: str) -> str:
        """Normalize code for more consistent comparison."""
        # Remove comments
        code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)

        # Normalize whitespace
        code = re.sub(r"\s+", " ", code)

        # Remove trailing/leading whitespace
        code = code.strip()

        # Normalize variable names to reduce superficial differences
        var_pattern = r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"
        vars_found = set(re.findall(var_pattern, code))

        normalized = code
        for idx, var in enumerate(sorted(vars_found)):
            normalized = re.sub(
                r"\b" + re.escape(var) + r"\b", f"var_{idx}", normalized
            )

        return normalized

    def check_implication(
        self, text1: str, text2: str, *args, **kwargs
    ) -> Dict[str, float]:
        """Check entailment between two code snippets with code-aware preprocessing."""
        # Normalize both code snippets
        norm_text1 = self.normalize_code(text1)
        norm_text2 = self.normalize_code(text2)

        logging.info(f"Normalized text1: {norm_text1}")
        logging.info(f"Normalized text2: {norm_text2}")

        # Use DeBERTa for the actual entailment check
        inputs = self.tokenizer(
            norm_text1, norm_text2, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)[0]

            # Return raw probabilities for use in semantic comparison
            return {
                "contradiction": probs[0].item(),
                "neutral": probs[1].item(),
                "entailment": probs[2].item(),
            }
