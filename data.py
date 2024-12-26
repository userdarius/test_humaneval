import os
import json
import datasets


def load_df(name, seed):
    user = "foodeei"

    test_df = None

    df = datasets.load_dataset(name)

    all_data = df["test"]

    def reformat_data(data):
        full_solution = data["test"]

        return {
            "question": f"Write a Python function that matches this signature and description:\n{data['prompt']}",
            "answers": {"text": [full_solution]},
            "context": data["test"],  # Store test cases as context
            "id": data["task_id"],
            "entry_point": data["entry_point"],
            "test_code": data["test"],  # Store original test code for evaluation
        }

    formatted_data = [reformat_data(d) for d in all_data]

    test_df = formatted_data

    return test_df


def get_dataset(name, seed):
    """Load and return a dataset by name and seed value."""
    print(f"Loading {name} dataset with seed {seed}")
    return load_df(name, seed)
