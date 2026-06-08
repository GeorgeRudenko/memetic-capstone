# Memetic Capstone

A demotivator generator powered by a fine-tuned Qwen2.5-7B model with LoRA.

## Project Overview

This project generates demotivational memes (demotivators) using a fine-tuned large language model. The system combines semantic template retrieval, text generation with a fine-tuned Qwen2.5-7B + LoRA model, and demotivator-style rendering (black background with white border and dual-caption layout).

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

How to Use

Open the web interface by running streamlit run app.py.
Enter a product or situation and describe the problem or pain point.
Click the "Generate demotivator" button.

memetic_capstone/
├── app.py                           # Streamlit web interface
├── meme_generator_local.py          # Core generation logic
├── meme_templates_clean.json        # Cleaned set of working meme templates
├── development_notebook.ipynb       # Development and experimentation notebook
├── checkpoints/
│   └── qwen_lora_memes_v1/          # Fine-tuned LoRA adapter
├── requirements.txt
└── README.md

Model Details

Base model: Qwen/Qwen2.5-7B-Instruct
Fine-tuning method: LoRA
Training data: Custom dataset of demotivator-style captions
Inference: Runs in bfloat16 precision (no quantization required on A100)

Deployment
Recommended Platform: Hugging Face Spaces

Create a new Space and select Streamlit as the SDK.
Upload the following files:
app.py
meme_generator_local.py
meme_templates_clean.json
requirements.txt
adapter_config.json and adapter_model.safetensors from the checkpoints/qwen_lora_memes_v1/ directory

Deploy the Space.

What Was Implemented

Collection and cleaning of a demotivator dataset
Fine-tuning of Qwen2.5-7B using LoRA for demotivator-style text generation
Semantic retrieval of meme templates using Sentence Transformers
Custom rendering engine for demotivator format (black background, white border, two-line caption)
Cleaning of the template database to include only templates with working image URLs
Development of a user-friendly Streamlit web interface

License
MIT License
