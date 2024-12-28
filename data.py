import datasets


def get_dataset(name, seed):
    """Load and return a dataset by name and seed value."""
    print(f"Loading {name} dataset with seed {seed}")

    df = datasets.load_dataset(name)
    return [
        {
            "question": f"Implement the following Python function without using placeholders like '# your code here', 'pass', or '...':\n\n{data['prompt']}",
            "answers": {"text": [data["test"]]},
            "context": data["test"],
            "id": data["task_id"],
            "entry_point": data["entry_point"],
            "test_code": data["test"],
        }
        for data in df["test"]
    ]
