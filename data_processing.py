import os
import json
import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import login
# For synthetic generation we'll use Groq or OpenAI compatible later

# Script for processing ImgFlip575K + synthetic augmentation for Memetic fine-tuning

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

def filter_memes(df, min_votes=100, min_views=1000):
    """Filter popular memes"""
    if 'votes' in df.columns and 'views' in df.columns:
        return df[(df['votes'] >= min_votes) & (df['views'] >= min_views)]
    return df

def preprocess_imgflip(dataset_dir):
    """Preprocess the dataset into structured format"""
    data = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    try:
                        meme = json.load(f)
                        template = meme.get('template_name', '') or meme.get('template', '')
                        texts = meme.get('texts', []) or meme.get('text', [])
                        title = meme.get('title', '')
                        votes = meme.get('votes', 0)
                        views = meme.get('views', 0)
                        
                        if texts and template:
                            data.append({
                                'template': template,
                                'meme_texts': texts,
                                'title': title,
                                'votes': votes,
                                'views': views,
                                'product': '',  # synthetic
                                'pain': '',     # synthetic
                                'emotion': ''   # synthetic
                            })
                    except Exception as e:
                        continue
    
    df = pd.DataFrame(data)
    print(f'Processed {len(df)} memes from ImgFlip')
    return df

def generate_synthetic_data(df_sample, num_samples=1000):
    """Use LLM (Groq/Qwen) to generate synthetic (product, pain, meme) examples"""
    # Placeholder - in real run use Groq client or vLLM
    print(f'Generating {num_samples} synthetic examples...')
    synthetic = []
    for i in range(min(num_samples, len(df_sample))):
        row = df_sample.iloc[i]
        synthetic.append({
            'product': 'example_product',
            'pain': 'example_pain',
            'meme_texts': row['meme_texts'],
            'template': row['template'],
            'emotion': 'frustration_to_motivation'
        })
    return pd.DataFrame(synthetic)

def save_to_hf_format(df, dataset_name='memetic-finetune'):
    """Save to Hugging Face Dataset format"""
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(f'data/processed/{dataset_name}')
    print(f'Saved to Hugging Face format in data/processed/{dataset_name}')
    # dataset.push_to_hub(dataset_name)  # uncomment after login

if __name__ == "__main__":
    dataset_dir = download_imgflip_dataset()
    df = preprocess_imgflip(dataset_dir)
    df = filter_memes(df)
    
    # Sample for synthetic generation
    sample_df = df.sample(min(5000, len(df))) if len(df) > 0 else df
    synthetic_df = generate_synthetic_data(sample_df, num_samples=2000)
    
    final_df = pd.concat([df.head(1000), synthetic_df], ignore_index=True)
    
    os.makedirs('data/processed', exist_ok=True)
    final_df.to_csv('data/processed/memes_processed.csv', index=False)
    save_to_hf_format(final_df)
    
    print('✅ Preprocessing and synthetic data generation complete!')