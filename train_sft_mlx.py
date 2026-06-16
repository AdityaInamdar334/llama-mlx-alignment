import os
import argparse
import sys
import subprocess
from pathlib import Path
from datasets import load_dataset

from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning (SFT) for Llama-3.2-1B-Instruct on Apple Silicon using MLX")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Hugging Face Model ID")
    parser.add_argument("--train_file", type=str, default="data/dpo_train.jsonl", help="Path to training JSONL file")
    parser.add_argument("--val_file", type=str, default="data/dpo_val.jsonl", help="Path to validation JSONL file")
    parser.add_argument("--output_dir", type=str, default="sft_model_mlx", help="Directory to save the trained model")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size per device")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout (0.0 is optimized in MLX)")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load base model in 4-bit quantized mode")
    parser.add_argument("--grad_checkpoint", action="store_true", default=False, help="Enable gradient checkpointing")
    parser.add_argument("--eval_steps", type=int, default=100, help="Steps between validation evaluations")
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

    args.train_file = resolve_path(args.train_file)
    args.val_file = resolve_path(args.val_file)
    args.output_dir = resolve_path(args.output_dir)
    
    print("=" * 60)
    print(f"Loading base model: {args.model_id}")
    print("=" * 60)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_gradient_checkpointing=args.grad_checkpoint,
    )
    
    print(f"Loading dataset from: {args.train_file} & {args.val_file}")
    # Load JSONL datasets
    dataset = load_dataset("json", data_files={"train": args.train_file, "val": args.val_file})
    train_dataset = dataset["train"]
    val_dataset = dataset["val"]
    
    # Format dataset for SFT
    def format_sft(example):
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["chosen"]}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}
    
    print("Formatting SFT training/validation datasets...")
    train_dataset = train_dataset.map(format_sft, desc="Formatting train dataset")
    val_dataset = val_dataset.map(format_sft, desc="Formatting val dataset")
    
    # Setup SFT Config
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        grad_checkpoint=args.grad_checkpoint,
    )
    
    # Run SFT trainer
    print("Starting SFT training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
        adapter_path=os.path.join(args.output_dir, "adapters")
    )
    
    trainer.train()
    
    print("Saving fine-tuned SFT adapters...")
    trainer.save_model(args.output_dir)
    
    # Run model fusion programmatically
    fused_dir = resolve_path("sft_model_mlx_merged")
    print(f"Fusing adapters into base model to create {fused_dir}...")
    
    # Run the fusion command
    # Use python interpreter in current virtual environment
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
        print(f"✓ Fused model successfully saved to: {fused_dir}")
    else:
        print(f"Error during model fusion (exit code {result.returncode}):")
        print(result.stdout)
        print(result.stderr)
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()