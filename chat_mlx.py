import os
import argparse
import sys
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

def main():
    parser = argparse.ArgumentParser(description="Chat with an MLX-aligned Llama model")
    parser.add_argument("--model_path", type=str, default="dpo_model_mlx_merged", help="Path to fused DPO/SFT model directory")
    args = parser.parse_args()

    model_path = args.model_path

    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' does not exist.")
        print("Please make sure you have run both train_sft_mlx.py and train_dpo_mlx.py first.")
        sys.exit(1)

    print(f"Loading model and tokenizer from '{model_path}'...")
    model, tokenizer = load(model_path)
    print("\n✓ Model loaded successfully!")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 60)

    history = []
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break
            if not user_input.strip():
                continue

            history.append({"role": "user", "content": user_input})
            
            # Use tokenizer's chat template
            formatted_prompt = tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True
            )
            
            print("\nAssistant: ", end="", flush=True)
            
            # Stream the generation
            response_text = ""
            for response in stream_generate(
                model,
                tokenizer,
                prompt=formatted_prompt,
                max_tokens=512,
                sampler=make_sampler(temp=0.7, top_p=0.9),
            ):
                print(response.text, end="", flush=True)
                response_text += response.text
            print() # Newline after response

            history.append({"role": "assistant", "content": response_text})

        except KeyboardInterrupt:
            print("\n\nSession ended. Goodbye!")
            break

if __name__ == "__main__":
    main()