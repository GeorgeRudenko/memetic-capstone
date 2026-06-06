import os
import json
import requests
from datasets import Dataset, load_dataset
import pandas as pd

# Script for processing ImgFlip575K dataset for Memetic fine-tuning

def download_imgflip_dataset():
    """Download or clone the ImgFlip575K dataset"""
    repo_url = "https://github.com/schesa/ImgFlip575K_Dataset.git"
    dataset_dir = "data/raw/imgflip"
    
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
        print("Cloning ImgFlip575K dataset...")
        os.system(f"git clone {repo_url} {dataset_dir}")
    else:
        print("Dataset already downloaded.")
    
    return dataset_dir

def preprocess_imgflip(dataset_dir):
    """Preprocess the dataset into format for fine-tuning"""
    # Assuming JSON or CSV files in the dataset
    data = []
    # Example processing logic - adjust based on actual structure
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    try:
                        meme = json.load(f)
                        # Extract relevant fields
                        template = meme.get('template', '')
                        texts = meme.get('texts', [])
                        if texts:
                            data.append({
                                'template': template,
                                'meme_texts': texts,
                                'product': '',  # to be filled synthetically
                                'pain': ''      # to be filled synthetically
                            })
                    except:
                        continue
    
    df = pd.DataFrame(data)
    print(f"Processed {len(df)} memes")
    return df

if __name__ == "__main__":
    dataset_dir = download_imgflip_dataset()
    df = preprocess_imgflip(dataset_dir)
    df.to_csv('data/processed/memes.csv', index=False)
    print("Preprocessing complete!")