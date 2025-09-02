import gzip
import json
import os

def explore_jgz_files(annotation_dir, category):
    """
    Explore .jgz files for a specific category and separate sequences by split (train/test).

    Args:
        annotation_dir (str): Path to the directory containing .jgz files.
        category (str): The category to explore (e.g., "apple").

    Returns:
        dict: A dictionary with keys "train" and "test", each containing a list of sequences.
    """
    sequences = {"train": [], "test": []}
    for split in ["train", "test"]:
        file_path = os.path.join(annotation_dir, f"{category}_{split}.jgz")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        try:
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
                sequences[split].extend(data.keys())
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return sequences

if __name__ == "__main__":
    annotation_dir = "data3/rogelio/co3d/co3d_anno"
    category = "apple"

    sequences = explore_jgz_files(annotation_dir, category)

    if sequences["train"]:
        print(f"Train sequences for category '{category}':")
        for seq in sequences["train"]:
            print(f"- {seq}")
    else:
        print(f"No train sequences found for category '{category}'.")

    if sequences["test"]:
        print(f"Test sequences for category '{category}':")
        for seq in sequences["test"]:
            print(f"- {seq}")
    else:
        print(f"No test sequences found for category '{category}'.")
