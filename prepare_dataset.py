import os
import argparse
from datasets import load_dataset

def parse_hh_rlhf_example(example):
    """
    Parses a single HH-RLHF example.
    """
    chosen_str = example["chosen"]
    rejected_str = example["rejected"]
    
    delimiter = "\n\nAssistant:"
    
    if delimiter in chosen_str:
        prompt, chosen_response = chosen_str.rsplit(delimiter, 1)
        prompt = prompt + delimiter
        
        if rejected_str.startswith(prompt):
            rejected_response = rejected_str[len(prompt):]
        else:
            _, rejected_response = rejected_str.rsplit(delimiter, 1)
            
        return {
            "prompt": prompt.strip(),
            "chosen": chosen_response.strip(),
            "rejected": rejected_response.strip()
        }
    else:
        return {
            "prompt": "",
            "chosen": chosen_str,
            "rejected": rejected_str
        }

def prepare_data(max_samples=None, test_size=0.1, output_dir="data"):
    print(f"Loading 'Anthropic/hh-rlhf' dataset from Hugging Face...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    
    if max_samples is not None and max_samples < len(dataset):
        print(f"Slicing dataset to first {max_samples} samples...")
        dataset = dataset.select(range(max_samples))
        
    print(f"Processing and formatting {len(dataset)} examples...")
    
    old_columns = dataset.column_names
    formatted_dataset = dataset.map(
        parse_hh_rlhf_example,
        remove_columns=old_columns,
        desc="Formatting dataset"
    )
    
    formatted_dataset = formatted_dataset.filter(
        lambda x: x["prompt"] is not None and len(x["prompt"]) > 0 and x["chosen"] is not None and x["rejected"] is not None,
        desc="Filtering empty records"
    )
    
    print(f"Successfully formatted {len(formatted_dataset)} examples.")
    
    split_dataset = formatted_dataset.train_test_split(test_size=test_size, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    
    print(f"Split data into {len(train_dataset)} train samples and {len(val_dataset)} validation samples.")
    
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "dpo_train.jsonl")
    val_path = os.path.join(output_dir, "dpo_val.jsonl")
    
    print(f"Saving train dataset to {train_path}...")
    train_dataset.to_json(train_path, orient="records", lines=True)
    
    print(f"Saving validation dataset to {val_path}...")
    val_dataset.to_json(val_path, orient="records", lines=True)
    
    print("✓ Dataset preparation and serialization complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and format Anthropic HH-RLHF for DPO training.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process (default: all).")
    parser.add_argument("--test_size", type=float, default=0.1, help="Validation split size fraction (default: 0.1).")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for saved JSONL files.")
    
    args = parser.parse_args()
    prepare_data(max_samples=args.max_samples, test_size=args.test_size, output_dir=args.output_dir)
