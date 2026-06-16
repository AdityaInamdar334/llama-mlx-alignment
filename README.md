

Below is a polished, well‑structured version of your README with badges, clearer headings, and no em dashes. You can replace your current `README.md` with this.

```markdown
# 🍏 Apple Silicon MLX LLaMA Alignment Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![MLX](https://img.shields.io/badge/MLX-latest-orange)](https://github.com/ml-explore/mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete, end‑to‑end pipeline for Supervised Fine‑Tuning (SFT) and Direct Preference Optimization (DPO) of LLaMA models, designed exclusively for Apple Silicon Macs using the `mlx` and `mlx-lm` libraries.

**Run modern LLM alignment on your Mac – no NVIDIA GPUs, no CUDA.**

## ✨ Features

- **100% Apple Silicon native** – Leverages the Neural Engine and unified memory via MLX.
- **Supervised Fine‑Tuning (SFT)** – Teach the model instruction following and conversation structure using LoRA adapters.
- **Direct Preference Optimization (DPO)** – Align model behavior with chosen/rejected pairs, avoiding the complexity of reward models.
- **Automatic weight fusion** – Merge LoRA adapters back into the base model for maximum inference speed.
- **Interactive chat CLI** – Real‑time streaming with adjustable temperature and sampler controls.
- **Easy deployment** – Export to Hugging Face Hub, run as an OpenAI‑compatible API, or convert to GGUF for Ollama.

## 📁 Repository Structure

| File | Purpose |
|------|---------|
| `prepare_dataset.py` | Downloads and formats the Anthropic HH‑RLHF dataset into MLX‑compatible `.jsonl` files. |
| `train_sft_mlx.py` | Runs Supervised Fine‑Tuning (SFT) to create an instruction‑tuned model. |
| `train_dpo_mlx.py` | Runs Direct Preference Optimization (DPO) on the SFT model. |
| `chat_mlx.py` | Lightweight CLI chat interface to test your final merged model. |
| `requirements.txt` | Python dependencies optimized for macOS. |

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
This creates a `data/` folder containing `dpo_train.jsonl` and `dpo_val.jsonl`.

### 3. Supervised Fine‑Tuning (SFT)
Train the model on the "chosen" responses to teach assistant‑like behavior:
```bash
python train_sft_mlx.py
```
*Output:* `sft_model_mlx_merged/`

### 4. Direct Preference Optimization (DPO)
Train the model to prefer "chosen" over "rejected" responses:
```bash
python train_dpo_mlx.py
```
*Output:* `dpo_model_mlx_merged/`

### 5. Chat with Your Model
```bash
python chat_mlx.py --model_path dpo_model_mlx_merged
```

## 🌐 Deploy Your Fine‑Tuned Model

Once you have the `dpo_model_mlx_merged` folder, here are three ways to host and share it.

### Option 1: Upload to Hugging Face Hub
```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload your-username/my-awesome-llama-model ./dpo_model_mlx_merged
```

### Option 2: Run as a Local OpenAI‑Compatible API
```bash
python -m mlx_lm.server --model dpo_model_mlx_merged
```
Now connect any OpenAI‑compatible client to `http://localhost:8080/v1/chat/completions`.

### Option 3: Export to GGUF and Use with Ollama
Convert the model to `.gguf` using `llama.cpp`, then import it into Ollama for use in apps like Chatbox or LM Studio.

## 📊 Example Results

*[You can add a screenshot or a short table of training loss, reward, or a qualitative comparison between base, SFT, and DPO models here.]*

## 🛠 Requirements

- **Hardware:** Apple Silicon Mac (M1, M2, M3, M4) with at least 16GB unified memory (32GB recommended for 7B+ models).
- **Software:** macOS 13.0+, Python 3.10+, and the packages listed in `requirements.txt`.

## 🤝 Contributing

Pull requests and issues are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Apple MLX team](https://github.com/ml-explore/mlx) for the excellent framework.
- [Anthropic](https://www.anthropic.com/) for the HH‑RLHF dataset.
- The open‑source community for making LLM alignment accessible.
```

---

## Next Steps

1. **Update your resume** with the project entry above.  
2. **Replace your current README** with the improved version.  
3. **Add a screenshot** of the chat CLI or a graph of training loss to the README (optional but impactful).  
4. **Push the update to GitHub** and consider posting a LinkedIn update about the project.  

Let me know if you also want a **LinkedIn post draft** or a **cover letter snippet** that mentions this new project.
