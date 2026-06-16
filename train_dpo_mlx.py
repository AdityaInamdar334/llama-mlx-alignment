import os
import argparse
import sys
import subprocess
from pathlib import Path
from datasets import load_dataset

from mlx_tune import FastLanguageModel, DPOTrainer, DPOConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Direct Preference Optimization (DPO) for Llama-3.2-1B-Instruct on Apple Silicon using MLX")
    parser.add_argument("--model_id", type=str, default="sft_model_mlx_merged", help="Hugging Face Model ID or local merged SFT model directory")
    parser.add_argument("--train_file", type=str, default="data/dpo_train.jsonl", help="Path to training JSONL file")
    parser.add_argument("--val_file", type=str, default="data/dpo_val.jsonl", help="Path to validation JSONL file")
    parser.add_argument("--output_dir", type=str, default="dpo_model_mlx", help="Directory to save the DPO model checkpoint")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size (set to 1 to enable prompt prefix sharing KV cache)")
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-7, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Maximum prompt length")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta scale factor")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout (0.0 is optimized in MLX)")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load base model in 4-bit quantized mode")
    parser.add_argument("--grad_checkpoint", action="store_true", default=False, help="Enable gradient checkpointing")
    parser.add_argument("--precompute_ref_logprobs", action="store_true", default=False, help="Precompute reference model logprobs (setting to False is recommended for speed)")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Checkpoint saving steps")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Resolve relative paths relative to script's parent directory
    script_dir = Path(__file__).resolve().parent
    
    def resolve_path(p):
        if p and not os.path.isabs(p):
            return str(script_dir / p)
        return p

    args.model_id = resolve_path(args.model_id)
    args.train_file = resolve_path(args.train_file)
    args.val_file = resolve_path(args.val_file)
    args.output_dir = resolve_path(args.output_dir)
    
    print("=" * 60)
    print(f"Loading merged SFT base model: {args.model_id}")
    print("=" * 60)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    
    print("Adding LoRA adapters for DPO...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_gradient_checkpointing=args.grad_checkpoint,
    )
    
    print(f"Loading dataset from: {args.train_file} & {args.val_file}")
    dataset = load_dataset("json", data_files={"train": args.train_file, "val": args.val_file})
    train_dataset = dataset["train"]
    val_dataset = dataset["val"]
    
    # Format dataset for DPO
    # Prompt is formatted as conversation user message up to the generation prompt.
    # Responses are formatted with the end-of-turn delimiter.
    def format_dpo(example):
        prompt_msg = [{"role": "user", "content": example["prompt"]}]
        prompt_formatted = tokenizer.apply_chat_template(prompt_msg, tokenize=False, add_generation_prompt=True)
        chosen_formatted = f"{example['chosen']}<|eot_id|>"
        rejected_formatted = f"{example['rejected']}<|eot_id|>"
        return {
            "prompt": prompt_formatted,
            "chosen": chosen_formatted,
            "rejected": rejected_formatted
        }
    
    print("Formatting preference datasets with instruct chat template...")
    train_dataset = train_dataset.map(format_dpo, desc="Formatting train DPO data")
    val_dataset = val_dataset.map(format_dpo, desc="Formatting val DPO data")
    
    # Setup DPO Config
    dpo_config = DPOConfig(
        beta=args.beta,
        precompute_ref_logprobs=args.precompute_ref_logprobs,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_seq_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
    )
    
    # Run DPO trainer
    print("Starting DPO preference alignment training...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None, # Automatically uses stop_gradient on base model parameters if None
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        args=dpo_config
    )
    
    trainer.train()
    
    # Run model fusion programmatically to get final aligned model
    fused_dir = resolve_path("dpo_model_mlx_merged")
    print(f"Fusing DPO adapters into SFT model to create final aligned model {fused_dir}...")
    
    python_bin = sys.executable
    cmd = [
        python_bin, "-m", "mlx_lm.fuse",
        "--model", args.model_id,
        "--save-path", fused_dir,
        "--adapter-path", os.path.join(args.output_dir, "adapters")
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ Final aligned DPO model successfully saved to: {fused_dir}")
    else:
        print(f"Error during final DPO model fusion (exit code {result.returncode}):")
        print(result.stdout)
        print(result.stderr)
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()