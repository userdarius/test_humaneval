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
        
        # Default stop sequences for Python code generation
        default_stops = ["```", "'''", '"""', "\ndef", "\nclass", "\n#", "\nif __name__"]
        if stop_sequences:
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]
            self.stop_sequences = stop_sequences + default_stops
        else:
            self.stop_sequences = default_stops

        # Token limit safeguard
        self.token_limit = self.target_model.config.max_position_embeddings
        
        # Configuration parameters
        self.min_acceptance_threshold = 0.1
        self.repetition_penalty = 1.2
        self.max_context_length = 20  # For repetition penalty window

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
                logging.info("  Number of attention heads: %s", config.num_attention_heads)
                logging.info("  Vocabulary size: %s", config.vocab_size)

    def _get_model_probabilities(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Get next token probabilities from model with improved temperature scaling and repetition penalty."""
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :].clone()  # Clone to avoid in-place modification issues
            
            # Apply repetition penalty to recent tokens
            if input_ids.size(1) > 1:
                recent_tokens = set(input_ids[0, -self.max_context_length:].tolist())
                for token_id in recent_tokens:
                    logits[0, token_id] /= self.repetition_penalty
            
            # Improve temperature scaling
            if temperature != 1.0:
                logits = logits / max(temperature, 1e-5)  # Prevent division by zero
                
            # Apply softmax with better numerical stability
            probs = F.softmax(logits, dim=-1)
            
            # Ensure valid probability distribution
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            return probs

    def _sample_token(self, probs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Sample a token from the probability distribution with temperature."""
        if temperature == 0:
            # Greedy sampling
            return torch.argmax(probs, dim=-1, keepdim=True)
        else:
            # Temperature sampling
            return torch.multinomial(probs, num_samples=1)

    def _check_stop_sequence(self, text: str, input_length: int) -> bool:
        """Check if any stop sequence is present in the generated portion."""
        generated_text = text[input_length:]
        return any(stop_seq in generated_text for stop_seq in self.stop_sequences)

    @torch.no_grad()
    def generate(
        self, 
        input_text: str, 
        temperature: float = 0.6, 
        return_full: bool = False,
        min_length: int = 50  # Minimum generation length before checking stop sequences
    ):
        """Generate text using speculative sampling with improved controls."""
        logging.info("Starting generation with input text: %s", input_text)
        logging.info("Temperature: %s", temperature)

        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        n_input_tokens = input_ids.size(1)
        
        # Initialize outputs
        outputs = SpeculativeOutput(
            sequences=input_ids.clone(),
            logits=torch.tensor([]),
            scores=[]
        )
        
        generation_step = 0
        consecutive_rejections = 0
        max_consecutive_rejections = 5

        while (outputs.sequences.shape[1] < n_input_tokens + self.max_new_tokens and
               consecutive_rejections < max_consecutive_rejections):
            
            generation_step += 1
            logging.info("\nGeneration Step %d:", generation_step)
            
            prefix_len = outputs.sequences.shape[1]
            current_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            
            # Generate draft sequence
            draft_sequence = outputs.sequences.clone()
            draft_probs = []
            draft_tokens = []

            logging.info("Generating draft sequence...")
            
            for i in range(self.gamma):
                probs = self._get_model_probabilities(self.approx_model, draft_sequence, temperature)
                next_token = self._sample_token(probs, temperature)
                draft_sequence = torch.cat((draft_sequence, next_token), dim=1)
                draft_probs.append(probs)
                draft_tokens.append(next_token)
                
                token_text = self.tokenizer.decode(next_token[0])
                logging.info("Draft token %d: %s (token_id: %d)", 
                           i+1, token_text, next_token.item())

            # Process with target model
            target_probs = []
            accepted_tokens = []

            for i in range(self.gamma):
                current_seq = draft_sequence[:, :prefix_len + i + 1]
                target_prob = self._get_model_probabilities(self.target_model, current_seq, temperature)
                target_probs.append(target_prob)

                j = draft_sequence[:, prefix_len + i]
                r = torch.rand(1)

                target_token_prob = target_prob[0, j]
                approx_token_prob = draft_probs[i][0, j]
                
                token_text = self.tokenizer.decode(j)
                logging.info("Token %d (%s):", i+1, token_text)
                logging.info("  Target probability: %.4f", target_token_prob.item())
                logging.info("  Approx probability: %.4f", approx_token_prob.item())
                
                # Improved acceptance criterion
                acceptance_ratio = target_token_prob / (approx_token_prob + 1e-10)
                if acceptance_ratio > self.min_acceptance_threshold:
                    acceptance_prob = min(1.0, acceptance_ratio)
                    if r < acceptance_prob:
                        accepted_tokens.append(j)
                        consecutive_rejections = 0
                        logging.info("  Token accepted (ratio: %.4f)", acceptance_ratio)
                        continue
                
                logging.info("  Token rejected (ratio: %.4f)", acceptance_ratio)
                consecutive_rejections += 1
                break

            # Update sequence
            if accepted_tokens:
                accepted_tensor = torch.cat(accepted_tokens).unsqueeze(0)
                outputs.sequences = torch.cat((outputs.sequences, accepted_tensor), dim=1)
                outputs.scores.extend(target_probs[:len(accepted_tokens)])
                logging.info("Accepted %d tokens", len(accepted_tokens))

            # Sample next token if needed
            if len(accepted_tokens) < self.gamma:
                target_prob = target_probs[len(accepted_tokens)]
                next_token = self._sample_token(target_prob, temperature)
                outputs.sequences = torch.cat((outputs.sequences, next_token), dim=1)
                outputs.scores.append(target_probs[len(accepted_tokens)])
                token_text = self.tokenizer.decode(next_token[0])
                logging.info("Sampled new token from target model: %s", token_text)

            # Check current output
            current_output = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            logging.info("\nCurrent full output: %s", current_output)
            
            # Only check stop sequences after minimum length
            if (len(current_output) - len(input_text)) > min_length:
                if self._check_stop_sequence(current_output, len(input_text)):
                    logging.info("Stop sequence found. Stopping generation.")
                    break

        return self._process_output(current_output, input_text, outputs, return_full)

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
        generated_portion = generated_text[len(input_text):].strip()
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
