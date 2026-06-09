import json
import torch
import random
import os
import gc
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
from sentence_transformers import SentenceTransformer
import numpy as np
import textwrap
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class MemeGenerator:
    def __init__(self, lora_path=".", templates_path="meme_templates_clean.json"):
        self.model = None
        self.tokenizer = None
        self.templates = None
        self.retriever = None
        # Фиксируем путь на текущую корневую директорию /app
        self.lora_path = lora_path
        self.templates_path = templates_path
        self.load_model()
        self.load_templates(self.templates_path)

    def load_model(self):
        # Принудительно чистим RAM от мусора перед загрузкой LLM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Если путь ведет на Google Диск — жестко переключаем на корень Hugging Face Space
        if "/content/drive" in self.lora_path or not os.path.exists(self.lora_path):
            self.lora_path = "."
        
        # Превращаем путь в абсолютный каноничный вид (/app)
        self.lora_path = os.path.abspath(self.lora_path)
                    
        print("Checking hardware configuration...")
        if torch.cuda.is_available():
            print("🚀 GPU detected! Loading Qwen2.5-7B + LoRA in bfloat16...")
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto"
            }
        else:
            print("⚠️ Running on CPU. Loading in float32 with low memory mode to prevent OOM...")
            model_kwargs = {
                "torch_dtype": torch.float32,
                "low_cpu_mem_usage": True,
                "device_map": "cpu"
            }
                        
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            trust_remote_code=True,
            **model_kwargs
        )
                        
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        self.tokenizer.pad_token = self.tokenizer.eos_token
                        
        print(f"Loading LoRA adapters from root directory: {self.lora_path}")
        
        # Проверяем физическое наличие конфигурационного файла адаптеров
        config_check_path = os.path.join(self.lora_path, "adapter_config.json")
        if not os.path.exists(config_check_path):
            raise FileNotFoundError(f"🚨 Critical Error: 'adapter_config.json' not found at {self.lora_path}!")

        # Загружаем адаптеры из корня приложения
        self.model = PeftModel.from_pretrained(
            base_model, 
            self.lora_path,
            is_trainable=False  # Жестко блокирует сетевую валидацию repo_id внутри hf_hub_download
        )
        self.model.eval()
        print("✅ Model loaded successfully!")

    def load_templates(self, path="meme_templates_clean.json"):
        with open(path, "r", encoding="utf-8") as f:
            self.templates = json.load(f)
        self.retriever = TemplateRetriever(self.templates)
        print(f"✅ Loaded {len(self.templates)} templates from {path}")

    def generate_meme(self, product: str, pain: str, output_path: str = "meme.png"):
        if self.model is None:
            self.load_model()
        if self.templates is None:
            self.load_templates(self.templates_path)
        
        query = f"{product} {pain}"
        candidates = self.retriever.get_relevant_templates(query, top_k=15)
                        
        random.shuffle(candidates)
                        
        best_key = random.choice(candidates[:10])
        template_info = self.templates[best_key]
        print(f"Chosen template: {best_key}")
        
        text = self._generate_text(product, pain, template_info)
        image_url = template_info.get("url", f"https://i.imgflip.com/{best_key}.jpg")
        self._render_meme(image_url, text, output_path)
        return text

    def _generate_text(self, product: str, pain: str, template_info: dict, max_new_tokens: int = 120):
        prompt = f"""Write a short, sharp demotivational meme caption for this situation:
Product: {product}
Problem: {pain}
Write exactly two lines separated by a newline character:
Line 1 (big text): A strong, punchy phrase (3-6 words)
Line 2 (small text): A sarcastic or ironic comment (5-10 words)
Only output the two lines, nothing else. Do not include labels like "first line:" or "second line:".
Example format:
DON'T PANIC
It's just a minor catastrophe
Meme caption:"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.15
            )
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                        
        if "Meme caption:" in text:
            text = text.split("Meme caption:")[-1].strip()
                        
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line.lower().startswith(('first line:', 'second line:', 'line 1:', 'line 2:', 'big text:', 'small text:')):
                line = line.split(':', 1)[-1].strip()
            if line and not line.lower().startswith(('example', 'format')):
                cleaned_lines.append(line)
                        
        if len(cleaned_lines) < 2 and len(cleaned_lines) == 1:
            words = cleaned_lines[0].split()
            if len(words) > 6:
                mid = len(words) // 2
                cleaned_lines = [' '.join(words[:mid]), ' '.join(words[mid:])]
                        
        result = '\n'.join(cleaned_lines[:2]) if cleaned_lines else text
        print(f"Generated text: {result}")
        return result

    def _render_meme(self, template_url: str, text: str, output_path: str, max_attempts: int = 3):
        original_img = None
        current_url = template_url
        attempts = 0
        while attempts < max_attempts and original_img is None:
            attempts += 1
            try:
                response = requests.get(current_url, timeout=6)
                if response.status_code == 200:
                    original_img = Image.open(BytesIO(response.content)).convert("RGB")
                else:
                    print(f"⚠️ Попытка {attempts}: статус {response.status_code} | {current_url}")
            except Exception as e:
                print(f"⚠️ Попытка {attempts}: ошибка загрузки | {current_url} | {e}")
            
            if original_img is None and attempts < max_attempts:
                new_key = random.choice(list(self.templates.keys()))
                current_url = self.templates[new_key].get("url", current_url)
                print(f"🔄 Пробуем другой шаблон: {new_key}")
                
        if original_img is None:
            print("❌ Не удалось загрузить ни одну картинку. Создаём заглушку.")
            original_img = Image.new("RGB", (700, 500), color="#222222")
            
        img_w, img_h = original_img.size
        margin = 50
        border_width = 6
        text_area_height = 170
        canvas_width = img_w + margin * 2 + border_width * 2
        canvas_height = img_h + margin * 2 + border_width * 2 + text_area_height
        
        canvas = Image.new("RGB", (canvas_width, canvas_height), "black")
        draw = ImageDraw.Draw(canvas)
        
        draw.rectangle(
            [margin, margin, canvas_width - margin, canvas_height - text_area_height - margin],
            outline="white",
            width=border_width
        )
        canvas.paste(original_img, (margin + border_width, margin + border_width))
        
        try:
            font_big = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 40)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans.ttf", 24)
        except:
            print("⚠️ Системные шрифты не найдены, используем дефолтные")
            font_big = ImageFont.load_default()
            font_small = ImageFont.load_default()
            
        lines = text.split('\n')
        if len(lines) >= 2:
            big_text = lines[0].strip().upper()
            small_text = ' '.join(lines[1:]).strip()
        else:
            words = text.split()
            if len(words) > 6:
                mid = len(words) // 2
                big_text = ' '.join(words[:mid]).upper()
                small_text = ' '.join(words[mid:])
            else:
                big_text = text.upper()
                small_text = ""
                
        big_text = big_text.replace("FIRST LINE:", "").replace("SECOND LINE:", "").strip()
        small_text = small_text.replace("FIRST LINE:", "").replace("SECOND LINE:", "").strip()
        
        if big_text:
            bbox = draw.textbbox((0, 0), big_text, font=font_big)
            text_w = bbox[2] - bbox[0]
            x = (canvas_width - text_w) // 2
            draw.text((x, canvas_height - text_area_height + 25), big_text, font=font_big, fill="white")
            
        if small_text:
            bbox2 = draw.textbbox((0, 0), small_text, font=font_small)
            text_w2 = bbox2[2] - bbox2[0]
            x2 = (canvas_width - text_w2) // 2
            draw.text((x2, canvas_height - 55), small_text, font=font_small, fill="white")
            
        canvas.save(output_path, quality=95)
        print(f"✅ Демотиватор сохранён: {output_path}")

class TemplateRetriever:
    def __init__(self, templates):
        self.templates = templates
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.descriptions = [t.get("description", t.get("name", "")) for t in templates.values()]
        self.keys = list(templates.keys())
        self.embeddings = self.model.encode(self.descriptions, normalize_embeddings=True)

    def get_relevant_templates(self, query, top_k=15):
        query_emb = self.model.encode([query], normalize_embeddings=True)
        similarities = np.dot(self.embeddings, query_emb.T).flatten()
                        
        noise = np.random.normal(0, 0.08, size=similarities.shape)
        similarities = similarities + noise
                        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        selected_keys = [self.keys[i] for i in top_indices]
                        
        random.shuffle(selected_keys)
        return selected_keys
