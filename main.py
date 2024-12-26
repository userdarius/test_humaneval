import torch
from data import get_dataset
from model import load_model_and_tokenizer
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def evaluate_model(model, tokenizer, dataset, num_samples=10):
    correct = 0
    total = 0

    for item in tqdm(dataset[:num_samples]):  # Limiting to num_samples for testing
        question = item["question"]

        # Prepare prompt
        prompt = f"""Write Python code to solve this problem. Only provide the code without any explanations.

{question}

Answer:"""

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        # Move input tensors to the same device as the model's first parameter
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt) :].strip()  # Remove prompt from response

        # Extract just the function implementation
        try:
            # Basic check - does it contain the entry point function?
            if item["entry_point"] in response:
                correct += 1
            total += 1

            logging.info(f"\nQuestion: {question[:100]}...")
            logging.info(f"Generated Response: {response[:100]}...")
            logging.info(
                f"Contains entry point '{item['entry_point']}': {item['entry_point'] in response}"
            )

        except Exception as e:
            logging.error(f"Error processing response: {e}")
            total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy


def main():
    # Model parameters
    model_name = "meta-llama/Llama-3.2-3b"  # You'll need HF access token

    # Load dataset
    dataset = get_dataset("openai_humaneval", seed=42)

    # Load model and tokenizer
    logging.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_name)

    # No need to manually move model to device since we're using device_map="auto"

    # Evaluate
    logging.info("Starting evaluation...")
    accuracy = evaluate_model(model, tokenizer, dataset)

    logging.info(f"\nFinal Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
