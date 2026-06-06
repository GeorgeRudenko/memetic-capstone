import torch
from unsloth import FastLanguageModel
from typing import List

class MemeticAI:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
        return cls._instance
    
    def _load_model(self):
        print("Loading model...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            gpu_memory_utilization=0.85,
        )
        FastLanguageModel.for_inference(self.model)
        print("Model loaded!")
    
    def generate(self, product: str, pain: str, num: int = 3) -> List[str]:
        prompt = f"""Create exactly {num} short funny memes.

Product: {product}
Pain: {pain}

Respond with one meme per line."""

        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=280, 
            temperature=0.9, 
            top_p=0.9, 
            repetition_penalty=1.2, 
            do_sample=True
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        memes = [line.strip() for line in response.split("\n") if line.strip() and len(line.strip()) > 8]
        return memes[:num] if memes else [f"{product} solves {pain} 🔥"] * num

# Singleton
memetic = MemeticAI()

def generate_memes(product: str, pain: str, num: int = 3) -> List[str]:
    return memetic.generate(product, pain, num)
