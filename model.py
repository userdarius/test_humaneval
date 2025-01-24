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
from kvcache_model import KVCacheModel
from utils import sample, max_fn


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


### Speculative sampling ###
@torch.no_grad()
def speculative_sampling(
    prefix: torch.Tensor,
    approx_model: torch.nn.Module,
    target_model: torch.nn.Module,
    max_len: int,
    gamma: int = 4,
    temperature: float = 1,
    top_k: int = 0,
    top_p: float = 0,
    verbose: bool = False,
    random_seed: int = None,
) -> tuple[torch.Tensor, list[float]]:  # Modified return type to include log probs
    """
    Google version Speculative Sampling with log probability tracking.
    Returns both the generated sequence and list of token log probabilities.
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len

    assert prefix.shape[0] == 1, "input batch size must be 1"
    assert approx_model.device == target_model.device

    device = target_model.device

    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)

    # Initialize list to store token log probabilities
    token_log_probs = []

    while prefix.shape[1] < T:
        prefix_len = prefix.shape[1]
        x = approx_model_cache.generate(prefix, gamma)
        _ = target_model_cache.generate(x, 1)

        n = prefix_len + gamma - 1

        for i in range(gamma):
            if random_seed:
                torch.manual_seed(random_seed)
            r = torch.rand(1, device=device)
            j = x[:, prefix_len + i]

            # Use normalized probabilities for acceptance/rejection
            target_prob = target_model_cache._prob_history[:, prefix_len + i - 1, j]
            approx_prob = approx_model_cache._prob_history[:, prefix_len + i - 1, j]

            if r > target_prob / approx_prob:
                n = prefix_len + i - 1
                break

            # Calculate log probability using log_softmax on raw logits
            logits = target_model_cache._logits_history[:, prefix_len + i - 1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_prob = log_probs[0, j].item()
            token_log_probs.append(token_log_prob)

            if verbose:
                print(f"approx guess accepted {j[0]}")
                print(f"log probability: {token_log_prob:.4f}")

        prefix = x[:, : n + 1]
        approx_model_cache.rollback(n + 1)

        if n < prefix_len + gamma - 1:
            # Rejection case
            logits = target_model_cache._logits_history[:, n, :]
            log_probs = F.log_softmax(logits, dim=-1)
            t = sample(
                max_fn(
                    target_model_cache._prob_history[:, n, :]
                    - approx_model_cache._prob_history[:, n, :]
                )
            )
            token_log_prob = log_probs[0, t].item()
            token_log_probs.append(token_log_prob)
            target_model_cache.rollback(n + 1)
        else:
            # All tokens accepted
            logits = target_model_cache._logits_history[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            t = sample(target_model_cache._prob_history[:, -1, :])
            token_log_prob = log_probs[0, t].item()
            token_log_probs.append(token_log_prob)

            target_model_cache.rollback(n + 2)

        prefix = torch.cat((prefix, t), dim=1)

    return prefix, token_log_probs


def load_approx_and_target_model_and_tokenizer(approx_model_name, target_model_name):
    approx_model = load_model(approx_model_name)
    target_model = load_model(target_model_name)
    tokenizer = load_tokenizer(target_model_name)
    return approx_model, target_model, tokenizer
