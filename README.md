# 🚀 Apple Silicon MLX LLaMA Alignment Pipeline

A complete, end-to-end Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) pipeline built entirely for Apple Silicon Macs using the `mlx` and `mlx-lm` libraries.

This project allows you to take a base LLaMA model and align it using modern reinforcement learning techniques (DPO) locally on your Mac, bypassing the need for expensive NVIDIA GPUs or complex CUDA setups.

## ✨ Features
*   **100% Apple Silicon Native**: Fully utilizes the Neural Engine and unified memory via Apple's MLX framework.
*   **Supervised Fine-Tuning (SFT)**: Teach the base model instruction-following and conversation structures using LoRA adapters.
*   **Direct Preference Optimization (DPO)**: Align the model's behavior using chosen/rejected pairs without the complexity of traditional RLHF reward models.
*   **Automatic Weight Fusion**: Seamlessly merge trained LoRA adapters back into the base model for maximum inference speed.
*   **Interactive Chat CLI**: A built-in terminal chat interface with real-time text streaming and temperature/sampler controls.

## 📁 Repository Structure
*   `prepare_dataset.py`: Downloads and formats the Anthropic HH-RLHF dataset into MLX-compatible `.jsonl` files.
*   `train_sft_mlx.py`: Runs Supervised Fine-Tuning to generate an intermediate instruction-tuned model.
*   `train_dpo_mlx.py`: Runs Direct Preference Optimization to align the SFT model.
*   `chat_mlx.py`: A lightweight CLI chat interface to test your final merged model.
*   `requirements.txt`: Python dependencies optimized for macOS.

## 🚀 Getting Started

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Data
Download and parse the RLHF preference dataset:
```bash
python prepare_dataset.py
```
*This will create a `data/` folder containing `dpo_train.jsonl` and `dpo_val.jsonl`.*

### 3. Supervised Fine-Tuning (SFT)
Train the model on the "chosen" responses to teach it how to behave as an assistant:
```bash
python train_sft_mlx.py
```
*Output: `sft_model_mlx_merged/`*

### 4. Direct Preference Optimization (DPO)
Train the model to prefer "chosen" responses over "rejected" ones:
```bash
python train_dpo_mlx.py
```
*Output: `dpo_model_mlx_merged/`*

### 5. Chat with Your Model
Interact with your fully aligned model in the terminal:
```bash
python chat_mlx.py --model_path dpo_model_mlx_merged
```

---

## 🌐 How to Host Your Fine-Tuned Model

Once you have your `dpo_model_mlx_merged` folder, you have several ways to host and share it:

### Option 1: Share on Hugging Face (Easiest)
You can upload your model to Hugging Face so anyone in the world can download it.
1. Create an account on huggingface.co
2. Run `pip install huggingface_hub` and `huggingface-cli login`
3. Push your model: `huggingface-cli upload your-username/my-awesome-llama-model ./dpo_model_mlx_merged`

### Option 2: Run as a Local API Endpoint
The MLX library includes a built-in server that mimics the OpenAI API format! You can start a server directly from your terminal:
```bash
python -m mlx_lm.server --model dpo_model_mlx_merged
```
You can now connect standard web apps or tools to `http://localhost:8080/v1/chat/completions`.

### Option 3: Export to Ollama
If you want to use your model in apps like Chatbox or LM Studio, you can convert it to a `.gguf` file using `llama.cpp` and import it into Ollama. 
