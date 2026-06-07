# Memetic Capstone

AI-Powered Meme Generator using Fine-tuned Model

Capstone Project for HSE Master's in Business Informatics

## Overview

This project develops a fine-tuned language model capable of generating modern, selling memes based on a product description and user pain point.

## Key Features

- Fine-tuned Qwen2.5-7B model for high-quality meme text generation
- Support for real-world meme datasets (MemeSafetyBench + ImgFlip templates)
- Clean Streamlit web interface
- Professional project structure ready for presentation

## Project Structure

```
memetic-capstone/
├── app.py                 # Streamlit web interface
├── inference.py           # Main generation logic
├── data_processing.py     # Dataset preparation
├── fine_tuning/
│   └── train.py           # Training script (Unsloth/LoRA)
├── notebooks/             # Experiment notebooks
├── data/                  # Datasets
├── presentation/          # Slides
├── requirements.txt
└── .gitignore
```

## Getting Started

```bash
git clone https://github.com/GeorgeRudenko/memetic-capstone.git
cd memetic-capstone
pip install -r requirements.txt
streamlit run app.py
```

## Model Training

The model is fine-tuned on a combination of:
- ImgFlip templates (clean images)
- MemeSafetyBench (real-world memes)
- High-quality synthetic data generated via powerful LLMs

## License

Educational project for HSE Capstone.