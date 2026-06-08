# Memetic Capstone
A meme generator powered by a fine-tuned Qwen2.5-7B model with LoRA. Creates product-focused memes in a demotivator-style format based on user-provided product descriptions and pain points.

## Project Overview
This project generates memes using a fine-tuned large language model. Given a product or situation and a related problem/pain point, the system:

1. Semantically retrieves the most relevant meme template
2. Generates a two-line meme caption using a fine-tuned Qwen2.5-7B + LoRA model
3. Renders the result in a demotivator-style format (black background with white border and dual-caption layout)

The result is a humorous, often sarcastic meme that highlights the irony or frustration of real-world product issues.

## How to Run

### Option 1: Google Colab (Recommended for quick demo)

1. Open the notebook: `development_notebook.ipynb` in Google Colab
2. Run all cells (`Runtime` → `Run all`)
3. In the last cell a public link will appear (e.g. `https://xxxx.gradio.live`)
4. Open the link to test the generator

### Option 2: Local Run

```bash
git clone https://github.com/GeorgeRudenko/memetic-capstone.git
cd memetic-capstone

pip install -r requirements.txt
python app.py
```

## Project Structure

```
memetic_capstone/
├── app.py                           # Gradio web interface
├── meme_generator_local.py          # Core generation logic
├── meme_templates_clean.json        # Cleaned set of working meme templates
├── development_notebook.ipynb       # Main development notebook
├── checkpoints/
│   └── qwen_lora_memes_v1/          # Fine-tuned LoRA adapter
├── requirements.txt
└── README.md
```

## Model Details

| Component            | Description                                      |
|----------------------|--------------------------------------------------|
| Base model           | Qwen/Qwen2.5-7B-Instruct                         |
| Fine-tuning method   | LoRA                                             |
| Training data        | Custom dataset of meme-style captions            |
| Inference precision  | bfloat16 (no quantization)                       |
| Template retrieval   | Sentence Transformers (all-MiniLM-L6-v2)         |

## Important: Cloning with Large Files

This repository uses **Git LFS** for the model file (`adapter_model.safetensors`).

```bash
git lfs install
git clone https://github.com/GeorgeRudenko/memetic-capstone.git
```

## Note on Deployment

An attempt was made to deploy the application to Hugging Face Spaces. However, due to difficulties installing PyTorch in the Spaces environment (especially with the large model), a stable deployment was not achieved within the available time.

For demonstration purposes, the project is run via Google Colab with Gradio's built-in public link sharing. This provides a working web interface without requiring complex infrastructure setup.

## What Was Implemented

- Collection and cleaning of a meme template dataset
- Fine-tuning of Qwen2.5-7B using LoRA for meme-style text generation
- Semantic retrieval of meme templates using Sentence Transformers
- Custom rendering engine for demotivator-style format (black background, white border, two-line caption)
- Cleaning of the template database to include only templates with working image URLs
- Development of a Gradio web interface

## License

MIT License