import os
import json
import datasets


def load_df(name, seed):
    user = "foodeei"

    test_df = None

    df = datasets.load_dataset(name)

    all_data = df["test"]

    def reformat_data(data):
        return {
            "question": (
                "Implement the following Python function without using placeholders like "
                "'# your code here', 'pass', or '...':\n\n"
                f"{data['prompt']}"
            ),
            "answers": {"text": [data["test"]]},
            "context": data["test"],
            "id": data["task_id"],
            "entry_point": data["entry_point"],
            "test_code": data["test"],
        }

    formatted_data = [reformat_data(d) for d in all_data]

    test_df = formatted_data

    return test_df


def get_dataset(name, seed):
    """Load and return a dataset by name and seed value."""
    print(f"Loading {name} dataset with seed {seed}")
    return load_df(name, seed)
