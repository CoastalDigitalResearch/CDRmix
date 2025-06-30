import os
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm

def slice_jsonl(input_path, output_dir, sample_size=5000, seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_lines = []
    print(f"Reading: {input_path}")
    with open(input_path, "r") as f:
        for line in f:
            all_lines.append(json.loads(line))

    print(f"Shuffling and sampling {sample_size} items from {len(all_lines)} total...")
    random.seed(seed)
    sample = random.sample(all_lines, min(sample_size, len(all_lines)))

    output_file = output_dir / "eval_set.jsonl"
    print(f"Writing to {output_file}")
    with open(output_file, "w") as out:
        for item in tqdm(sample):
            json.dump({"text": item.get("text", "")}, out)
            out.write("\n")

    print("✅ Done. Evaluation set written to:", output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to CommonPile .jsonl")
    parser.add_argument("--output_dir", required=True, help="Where to save eval dataset")
    parser.add_argument("--sample_size", type=int, default=5000, help="How many samples to keep")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    slice_jsonl(args.input, args.output_dir, args.sample_size, args.seed)
