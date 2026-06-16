# Apple Silicon MLX LLaMA Alignment Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-orange)](https://github.com/ml-explore/mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete end-to-end alignment pipeline for training and optimizing LLaMA-based language models on Apple Silicon hardware using MLX and MLX-LM.

This project demonstrates modern large language model alignment techniques, including Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO), without requiring NVIDIA GPUs or CUDA.

---

## Overview

The pipeline consists of:

1. Dataset preparation and preprocessing
2. Supervised Fine-Tuning (SFT) using LoRA adapters
3. Direct Preference Optimization (DPO) for preference alignment
4. LoRA weight merging for efficient inference
5. Interactive model evaluation through a command-line chat interface

The entire workflow is optimized for Apple Silicon devices and leverages unified memory through the MLX framework.

---

## Key Features

* Native Apple Silicon training and inference with MLX
* Parameter-efficient fine-tuning using LoRA
* Direct Preference Optimization for alignment without reward modeling
* Automatic adapter merging for deployment-ready checkpoints
* Streaming chat interface with configurable generation settings
* Multiple deployment options including Hugging Face, OpenAI-compatible APIs, and GGUF export

---

## Repository Structure

| File                 | Description                                                                         |
| -------------------- | ----------------------------------------------------------------------------------- |
| `prepare_dataset.py` | Downloads and formats the Anthropic HH-RLHF dataset into MLX-compatible JSONL files |
| `train_sft_mlx.py`   | Performs Supervised Fine-Tuning using LoRA adapters                                 |
| `train_dpo_mlx.py`   | Performs Direct Preference Optimization on the SFT model                            |
| `chat_mlx.py`        | Interactive CLI for model evaluation                                                |
| `requirements.txt`   | Python dependencies                                                                 |

---

## Installation

### Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Dataset Preparation

Download and preprocess the preference dataset:

```bash
python prepare_dataset.py
```

Generated files:

```text
data/
├── dpo_train.jsonl
└── dpo_val.jsonl
```

---

## Supervised Fine-Tuning (SFT)

Train the base model on preferred responses to establish instruction-following behavior.

```bash
python train_sft_mlx.py
```

Output:

```text
sft_model_mlx_merged/
```

---

## Direct Preference Optimization (DPO)

Further align the model using preference pairs consisting of chosen and rejected responses.

```bash
python train_dpo_mlx.py
```

Output:

```text
dpo_model_mlx_merged/
```

---

## Inference

Launch the interactive chat interface:

```bash
python chat_mlx.py --model_path dpo_model_mlx_merged
```

Example:

```text
User: Explain reinforcement learning in simple terms.

Assistant:
Reinforcement learning is a machine learning technique in which an agent learns by interacting with an environment and receiving feedback in the form of rewards...
```

---

## Deployment Options

### Hugging Face Hub

```bash
pip install huggingface_hub

huggingface-cli login

huggingface-cli upload \
your-username/apple-silicon-llama-alignment \
./dpo_model_mlx_merged
```

### OpenAI-Compatible API Server

```bash
python -m mlx_lm.server \
--model dpo_model_mlx_merged
```

Endpoint:

```text
http://localhost:8080/v1/chat/completions
```

### GGUF Export

Convert the merged model to GGUF format using `llama.cpp` and deploy with:

* Ollama
* LM Studio
* Open WebUI
* Chatbox

---

## Results

Suggested metrics to include:

| Metric              | Base Model | SFT | DPO |
| ------------------- | ---------- | --- | --- |
| Validation Loss     | -          | -   | -   |
| Preference Accuracy | -          | -   | -   |
| Win Rate            | -          | -   | -   |

You may also include:

* Training loss curves
* Validation metrics
* Inference screenshots
* Qualitative response comparisons

---

## System Requirements

### Hardware

* Apple Silicon Mac (M1, M2, M3, or M4)
* Minimum 16 GB unified memory
* Recommended 32 GB+ for 7B parameter models

### Software

* macOS 13.0 or later
* Python 3.10+
* MLX
* MLX-LM

---

## Technical Stack

* Python
* MLX
* MLX-LM
* LoRA
* Direct Preference Optimization (DPO)
* Hugging Face Datasets
* LLaMA Models

---

## Future Improvements

* Support for larger LLaMA variants
* Quantized training workflows
* Multi-GPU distributed support
* Evaluation benchmark suite
* Automated experiment tracking

---

## License

This project is released under the MIT License.

See the `LICENSE` file for additional information.

---

## Acknowledgments

* Apple MLX Team for the MLX framework
* Anthropic for the HH-RLHF dataset
* The open-source machine learning community
