# SLM-fine-tuning
---

# 📚 Gemma 3-270M Fine-Tuning for Food Information Extraction

This repository contains a Jupyter Notebook (`fine-tuning.ipynb`) demonstrating the end-to-end process of fine-tuning **Google's Gemma 3-270M (Instruction Tuned)** model. The goal is to distill the capabilities of a large 120B parameter model (GPT-OSS-120B) into a lightweight 270M parameter Small Language Model (SLM) for specialized text extraction.

---

## 📁 Project Overview

The project focuses on **Knowledge Distillation**. We use a dataset of image captions describing food and drinks and train the model to output **condensed structured data** rather than generic conversational text.

### Key Features:

* **Model:** `google/gemma-3-270m-it` (Instruction Tuned).
* **Dataset:** `mrdbourke/FoodExtract-1k` (Hugging Face).
* **Technique:** Supervised Fine-Tuning (SFT) using the `trl` library.
* **Optimization:** Mixed precision training (`bfloat16`) and gradient checkpointing to fit on a single Tesla T4 GPU (16GB).

---

## 🛠️ Repository Structure

* **`fine-tuning.ipynb`**: The main workflow notebook including:
* GPU Memory environment checks.
* Base model loading and pipeline testing.
* Dataset transformation into LLM "Conversation" style.
* Prompt engineering (Few-shot learning) comparison.
* Model training using `SFTTrainer`.
* Evaluation and inference with the fine-tuned checkpoint.


* **`checkpoints/`**: Directory for storing model weights at each epoch (created during training).

---

## 🚀 How to Use

### 1. Requirements

To run this notebook, you will need a GPU with at least 16GB of VRAM (e.g., NVIDIA T4, L4, or A100). Install the dependencies using the first cells in the notebook:

```bash
pip install trl gradio accelerate transformers datasets wandb

```

### 2. Training Workflow

1. **Format Data:** The notebook converts raw food descriptions into a `user/system` message format.
2. **SFT Configuration:**
* Learning Rate: `5e-5`
* Batch Size: `4`
* Epochs: `3`
* Max Length: `512`


3. **Run Training:** The `SFTTrainer` processes 1,136 training samples.

### 3. Inference

After training, load the local checkpoint to perform extraction:

```python
from transformers import pipeline
# Load the fine-tuned weights from your checkpoint folder
loaded_model_pipeline = pipeline("text-generation", model="./checkpoints/checkpoint-852")

```

---

## 📊 Results & Performance

* **Size Efficiency:** The fine-tuned **Gemma 3-270M** is approximately **444x smaller** than the teacher model (120B parameters) used to label the original data.
* **Task Success:** While the base model originally responded with generic conversational text, the fine-tuned version successfully extracts specific food/drink items and tags (e.g., `np` for nutrition panel, `il` for ingredient list) in a condensed format.

| Epoch | Training Loss | Validation Loss |
| --- | --- | --- |
| 1 | 2.1543 | 2.2339 |
| 2 | 1.2549 | 2.2818 |
| 3 | 1.0176 | 2.4629 |

---

## 🛠️ Technical Details

* **Core Model:** Gemma 3 270M Flash.
* **Architecture:** Identical to Gemini but at a smaller scale.
* **Quantization/Precision:** `torch.bfloat16`.
* **Image/Text Modalities:** While this notebook focuses on text-to-text extraction, the model architecture supports natively generated audio and image context.

---

