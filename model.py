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


### Speculative sampling model ###
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging


@dataclass
class SpeculativeOutput:
    sequences: torch.Tensor
    logits: torch.Tensor
    scores: List[torch.Tensor]


class SpeculativeSamplingModel:
    def __init__(
        self,
        approx_model_name: str,
        target_model_name: str,
        stop_sequences: Union[List[str], str] = None,
        max_new_tokens: int = 1024,
    ):
        # Set up logging
        logging.info("%s", f"\n{'='*50}")
        logging.info("Initializing SpeculativeSamplingModel:")
        logging.info("Target model: %s", target_model_name)
        logging.info("Approximation (Draft) model: %s", approx_model_name)

        # Initialize device

        # Initialize models and tokenizers
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name, device_map="auto"
        )
        self.approx_model = AutoModelForCausalLM.from_pretrained(
            approx_model_name, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)

        # Set parameters
        self.max_new_tokens = max_new_tokens
        self.gamma = 4  # Number of tokens to generate speculatively
        self.stop_sequences = (
            stop_sequences
            if isinstance(stop_sequences, list)
            else [stop_sequences] if stop_sequences else None
        )

        # Token limit safeguard
        self.token_limit = self.target_model.config.max_position_embeddings

        # Log model architectures
        self._log_model_info()

    def _log_model_info(self):
        """Log detailed information about both models"""
        for name, model in [
            ("Target", self.target_model),
            ("Approximation", self.approx_model),
        ]:
            logging.info(f"\n{name} Model Architecture:")
            logging.info("Model type: %s", type(model).__name__)
            logging.info(
                "Number of parameters: %s",
                f"{sum(p.numel() for p in model.parameters()):,}",
            )

            if hasattr(model, "config"):
                config = model.config
                logging.info("Configuration:")
                logging.info("  Hidden size: %s", config.hidden_size)
                logging.info("  Number of layers: %s", config.num_hidden_layers)
                logging.info(
                    "  Number of attention heads: %s", config.num_attention_heads
                )
                logging.info("  Vocabulary size: %s", config.vocab_size)

        if torch.cuda.is_available():
            logging.info("\nGPU Memory Usage:")
            logging.info("Allocated: %.2f MB", torch.cuda.memory_allocated() / 1024**2)
            logging.info("Cached: %.2f MB", torch.cuda.memory_reserved() / 1024**2)

    def _get_model_probabilities(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Get next token probabilities from model."""
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]  # Get logits for last token
            probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
            return probs

    def _sample_token(self, probs: torch.Tensor) -> torch.Tensor:
        """Sample a token from the probability distribution."""
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(
        self, input_text: str, temperature: float, return_full: bool = False
    ):
        """Generate text using speculative sampling."""
        logging.info("Starting prediction with temperature %s", temperature)

        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        n_input_tokens = input_ids.size(1)

        # Initialize output structure
        outputs = SpeculativeOutput(
            sequences=input_ids.clone(), logits=torch.tensor([]), scores=[]
        )
        # Generation loop
        while outputs.sequences.shape[1] < n_input_tokens + self.max_new_tokens:
            prefix_len = outputs.sequences.shape[1]

            # Generate draft sequence using approximation model
            draft_sequence = outputs.sequences.clone()
            draft_probs = []

            logging.info("Generating draft sequence")

            for _ in range(self.gamma):
                probs = self._get_model_probabilities(
                    self.approx_model, draft_sequence, temperature
                )
                next_token = self._sample_token(probs)
                draft_sequence = torch.cat((draft_sequence, next_token), dim=1)
                draft_probs.append(probs)

            # Get target model probabilities for the draft sequence
            target_probs = []
            accepted_tokens = []

            for i in range(self.gamma):
                current_seq = draft_sequence[:, : prefix_len + i + 1]
                target_prob = self._get_model_probabilities(
                    self.target_model, current_seq, temperature
                )
                target_probs.append(target_prob)

                # Accept/reject step
                j = draft_sequence[:, prefix_len + i]
                r = torch.rand(1)

                logging.info("Accept/reject step")
                

                target_token_prob = target_prob[0, j]
                approx_token_prob = draft_probs[i][0, j]
                logging.info("Target token probability: %s", target_token_prob)
                logging.info("Approx token probability: %s", approx_token_prob)

                if r > target_token_prob / approx_token_prob:
                    break

                accepted_tokens.append(j)

            # Update sequence with accepted tokens
            if accepted_tokens:
                accepted_tensor = torch.cat(accepted_tokens).unsqueeze(0)
                outputs.sequences = torch.cat(
                    (outputs.sequences, accepted_tensor), dim=1
                )

            # Sample next token if needed
            if len(accepted_tokens) < self.gamma:
                # Use target model to sample next token
                target_prob = target_probs[len(accepted_tokens)]
                next_token = self._sample_token(target_prob)
                outputs.sequences = torch.cat((outputs.sequences, next_token), dim=1)

            outputs.scores.extend(target_probs[: len(accepted_tokens) + 1])

            # Check for stop sequences
            generated_text = self.tokenizer.decode(
                outputs.sequences[0], skip_special_tokens=True
            )
            if self.stop_sequences:
                for stop_seq in self.stop_sequences:
                    if stop_seq in generated_text[len(input_text) :]:
                        return self._process_output(
                            generated_text, input_text, outputs, return_full
                        )

        return self._process_output(generated_text, input_text, outputs, return_full)

    def _process_output(
        self,
        generated_text: str,
        input_text: str,
        outputs: SpeculativeOutput,
        return_full: bool,
    ):
        """Process the generated output and return appropriate format."""
        if return_full:
            return generated_text

        # Extract generated portion
        generated_portion = generated_text[len(input_text) :].strip()

        logging.info("Generated portion: %s", generated_portion)

        # Compute log probabilities for generated tokens
        log_probs = []
        for score in outputs.scores:
            log_probs.append(torch.log(score).max().item())

        return generated_portion, log_probs


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
