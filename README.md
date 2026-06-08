# Memetic Capstone
A meme generator powered by a fine-tuned Qwen2.5-7B model with LoRA. Creates product-focused memes in a demotivator-style format based on user-provided product descriptions and pain points.

## Project Overview
This project generates memes using a fine-tuned large language model. Given a product or situation and a related problem/pain point, the system:

1. Semantically retrieves the most relevant meme template
2. Generates a two-line meme caption using a fine-tuned Qwen2.5-7B + LoRA model
3. Renders the result in a demotivator-style format (black background with white border and dual-caption layout)

The result is a humorous, often sarcastic meme that highlights the irony or frustration of real-world product issues.

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to Use

1. Open the web interface by running `streamlit run app.py`
2. Enter a product or situation (e.g., "Jira", "Zoom", "Google Docs")
3. Describe the problem or pain point (e.g., "backlog task accidentally got into to-do")
4. Click the "Generate meme" button
5. View and download your generated meme

## Project Structure

```
memetic_capstone/
├── app.py                           # Streamlit web interface
├── meme_generator_local.py          # Core generation logic
├── meme_templates_clean.json        # Cleaned set of working meme templates
├── development_notebook.ipynb       # Development and experimentation notebook
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
| Inference precision  | bfloat16 (no quantization required on A100)      |
| Template retrieval   | Sentence Transformers (all-MiniLM-L6-v2)         |

## Deployment

**Recommended Platform:** Hugging Face Spaces

1. Create a new Space and select **Streamlit** as the SDK
2. Upload the following files:
   - `app.py`
   - `meme_generator_local.py`
   - `meme_templates_clean.json`
   - `requirements.txt`
   - `adapter_config.json` and `adapter_model.safetensors` from `checkpoints/qwen_lora_memes_v1/`
3. Deploy the Space

## What Was Implemented

- Collection and cleaning of a meme template dataset
- Fine-tuning of Qwen2.5-7B using LoRA for meme-style text generation
- Semantic retrieval of meme templates using Sentence Transformers
- Custom rendering engine for demotivator-style format (black background, white border, two-line caption)
- Cleaning of the template database to include only templates with working image URLs
- Development of a user-friendly Streamlit web interface

## License

MIT License