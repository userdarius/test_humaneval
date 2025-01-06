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
from typing import Optional, List, Dict, Tuple
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

### Branching Model ###
def get_topk_next_tokens(
    model: AutoModelForCausalLM, inputs: Dict[str, torch.Tensor], num_branches: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get the top k most likely next tokens and their probabilities.
    Also returns the log probabilities.
    """
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True, temperature=0.4)
        next_token_logits = outputs.logits[:, -1, :]

    log_probs = torch.log_softmax(next_token_logits, dim=-1)  # Get log probabilities
    probabilities = torch.softmax(next_token_logits, dim=-1)
    topk_values, topk_indices = torch.topk(probabilities, num_branches)
    topk_logprobs = torch.gather(
        log_probs, -1, topk_indices
    )  # Get corresponding log probs

    return topk_values, topk_indices, topk_logprobs


def generate_single_branch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_length: int,
    inputs: Dict[str, torch.Tensor],
) -> Tuple[str, float, float]:
    response_tokens = [inputs["input_ids"][0, -1].item()]
    prob_diffs = []
    sequence_logprob = 0.0  # Track sequence log probability

    print(f"Starting with initial token: '{tokenizer.decode([response_tokens[0]])}'")

    for step in range(max_length):
        topk_values, topk_indices, topk_logprobs = get_topk_next_tokens(
            model, inputs, num_branches=10
        )

        next_token = topk_indices[0, 0].item()
        next_token_text = tokenizer.decode([next_token])

        # Add log probability of chosen token
        sequence_logprob += topk_logprobs[0, 0].item()

        current_text = tokenizer.decode(response_tokens + [next_token])
        print(f"Step {step}: Token {next_token} -> '{next_token_text}'")
        print(f"Current text: '{current_text}'")

        # Check stopping conditions
        if any(
            stop in next_token_text for stop in [".", "\n", "Explanation:", "Answer:"]
        ):
            if next_token_text.strip() == "." and not any(
                stop in current_text for stop in [".", "\n", "Explanation:", "Answer:"]
            ):
                response_tokens.append(next_token)
                sequence_logprob += topk_logprobs[0, 0].item()
            break

        # Regular token processing
        prob_diff = (topk_values[0, 0] - topk_values[0, 1]).item()
        response_tokens.append(next_token)
        prob_diffs.append(prob_diff)

        # Update inputs
        next_token_tensor = torch.tensor(
            [[next_token]], device=inputs["input_ids"].device
        )
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token_tensor], dim=1)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = torch.cat(
                [
                    inputs["attention_mask"],
                    torch.ones((1, 1), device=inputs["attention_mask"].device),
                ],
                dim=1,
            )

    # Convert token IDs to text at the end
    generated_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    avg_prob_diff = sum(prob_diffs) / len(prob_diffs) if prob_diffs else 0

    # Normalize log probability by sequence length
    normalized_logprob = (
        sequence_logprob / len(response_tokens) if response_tokens else 0
    )

    return generated_text.strip(), avg_prob_diff, normalized_logprob


def generate_branching_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int,
    num_branches: int,
) -> List[Tuple[str, float, float]]:
    """
    Generate multiple responses by exploring different initial tokens.
    Returns tuples of (text, confidence_score, log_probability)
    """
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Get initial top k tokens
    topk_values, topk_indices, topk_logprobs = get_topk_next_tokens(
        model, inputs, num_branches
    )  # Updated to unpack 3 values

    # Log initial token choices
    for k in range(num_branches):
        token_text = tokenizer.decode(topk_indices[0, k])
        print(
            f"Initial token {k+1}: {token_text} (prob: {topk_values[0,k]:.4f}, log_prob: {topk_logprobs[0,k]:.4f})"
        )

    responses = []
    for k in range(num_branches):
        print(f"\nStarting branch {k+1}")
        first_token = topk_indices[:, k : k + 1]
        first_token_text = tokenizer.decode(first_token[0])
        print(f"First token: '{first_token_text}'")
        # if first token is a stop token, skip this branch
        if any(
            stop in first_token_text
            for stop in [".", "\n", "Explanation:", "Answer:", r" \ ", "\\"]
        ):
            print(f"Skipping branch {k+1} because it starts with a stop token")
            continue

        # Create a new branch starting with the k-th most likely token
        branch_inputs = {
            "input_ids": torch.cat(
                [inputs["input_ids"], topk_indices[:, k : k + 1]], dim=1
            ),
            "attention_mask": (
                torch.cat(
                    [
                        inputs["attention_mask"],
                        torch.ones((1, 1), device=inputs["attention_mask"].device),
                    ],
                    dim=1,
                )
                if "attention_mask" in inputs
                else None
            ),
        }

        # Generate the rest of the response for this branch
        generated_text, confidence_score, log_prob = generate_single_branch(
            model, tokenizer, max_length, branch_inputs
        )

        responses.append((generated_text, confidence_score, log_prob))

    print("\nAll branches complete\n")
    return responses
