import os
import json
import pandas as pd
from datasets import Dataset
import random
from tqdm import tqdm

# For synthetic generation - we'll use Groq or OpenAI-compatible API later
# os.environ["GROQ_API_KEY"] = "your_key_here"

def download_imgflip_dataset():
    """Download or clone the ImgFlip575K dataset"""
    repo_url = "https://github.com/schesa/ImgFlip575K_Dataset.git"
    dataset_dir = "data/raw/imgflip"
    
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
        print("Cloning ImgFlip575K dataset... (this may take a few minutes)")
        os.system(f"git clone {repo_url} {dataset_dir} --depth 1")
    else:
        print("Dataset already downloaded.")
    
    return dataset_dir

def load_and_filter_memes(dataset_dir, min_votes=50):
    """Load JSON files and filter high-quality memes"""
    data = []
    json_files = []
    
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    print(f"Found {len(json_files)} JSON files. Processing...")
    
    for json_path in tqdm(json_files[:1000]):  # Limit to speed up initial run
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                meme = json.load(f)
                
                template_name = meme.get('template_name') or meme.get('template', '')
                texts = meme.get('texts') or meme.get('text_boxes', []) or meme.get('text', [])
                votes = meme.get('votes') or meme.get('up_votes', 0)
                views = meme.get('views', 0)
                
                if texts and len(texts) >= 1 and votes >= min_votes:
                    data.append({
                        'template': template_name,
                        'meme_texts': texts[:3],
                        'votes': votes,
                        'views': views,
                        'product': None,
                        'pain': None,
                        'emotion': None
                    })
        except:
            continue
    
    df = pd.DataFrame(data)
    print(f"Filtered to {len(df)} high-quality memes")
    return df

def generate_synthetic_data(df, num_samples=5000):
    """Generate synthetic (product, pain) using placeholder. Replace with real LLM call (Groq/Qwen)."""
    print(f"Generating {num_samples} synthetic examples...")
    
    products = ['gym', 'fitness app', 'coffee shop', 'online course', 'crypto wallet', 'language learning app', 'meal delivery', 'meditation app']
    pains = ['no motivation to workout', 'tired after work', 'too expensive', 'boring lessons', 'hard to track expenses', 'forgetting words', 'no time to cook', 'stress and anxiety']
    emotions = ['frustration', 'irony', 'motivation', 'relatable']
    
    synthetic = []
    sample_df = df.sample(n=min(num_samples, len(df)))
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        product = random.choice(products)
        pain = random.choice(pains)
        emotion = random.choice(emotions)
        
        synthetic.append({
            'template': row['template'],
            'meme_texts': row['meme_texts'],
            'product': product,
            'pain': pain,
            'emotion': emotion,
            'original_votes': row.get('votes', 0)
        })
    
    return pd.DataFrame(synthetic)

def save_to_hf_format(df, output_dir="data/processed"):
    """Save as Hugging Face Dataset"""
    os.makedirs(output_dir, exist_ok=True)
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(output_dir)
    print(f"✅ Saved {len(df)} examples to {output_dir}")
    return dataset

if __name__ == "__main__":
    # Run processing
    dataset_dir = download_imgflip_dataset()
    df = load_and_filter_memes(dataset_dir)
    synthetic_df = generate_synthetic_data(df, num_samples=8000)
    
    dataset = save_to_hf_format(synthetic_df)
    synthetic_df.to_csv('data/processed/memes_synthetic.csv', index=False)
    
    print("🎉 Data processing complete! Dataset ready for fine-tuning with Qwen2.5.")