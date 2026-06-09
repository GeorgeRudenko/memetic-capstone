import sys
import subprocess
import os

# Глобально отключаем SSR для предотвращения проблем с прокси
os.environ["GRADIO_SSR"] = "False"

# Force correct isolated installation for CPU torch and standard libraries
try:
    import torch
    import gradio as gr
except ImportError:
    print("📦 Bootstrapping lightweight CPU environment manually...")
    try:
        # Step 1: Install CPU-only PyTorch from the dedicated PyTorch wheels index
        print("-> Installing CPU-only PyTorch...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ])
        
        # Step 2: Install UI and NLP libraries from the global PyPI registry
        # Фиксируем стабильную версию gradio, чтобы обойти баг парсера схем
        print("-> Installing NLP and UI libraries...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "gradio==5.8.0",
            "transformers>=4.45.0,<4.49.0",
            "peft>=0.12.0",
            "sentence-transformers>=3.0.0",
            "Pillow",
            "requests",
            "numpy",
            "accelerate>=0.26.0",
            "huggingface_hub==0.25.2"
        ])
        print("✅ All CPU dependencies successfully installed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed with exit code {e.returncode}")
        raise e

# Core imports
import gradio as gr
from meme_generator_local import MemeGenerator

# Initialize the generator (loads model weights and templates internally)
generator = MemeGenerator()

# Явно типизируем входные аргументы для стабильности парсера Gradio
def generate_demotivator(product: str, pain: str):
    if not product or not pain:
        return None, "Please fill in both fields."
    
    output_path = "generated_meme.png"
    text = generator.generate_meme(product.strip(), pain.strip(), output_path)
    return output_path, text

# Gradio Interface
with gr.Blocks(title="Memetic Generator") as demo:
    gr.Markdown("# 🧠 Memetic Generator")
    gr.Markdown("Demotivator meme generator powered by fine-tuned Qwen2.5-7B + LoRA")
    
    with gr.Row():
        product = gr.Textbox(label="Product / Situation", placeholder="Jira, Slack, Notion...")
        pain = gr.Textbox(label="Pain / Problem", placeholder="backlog task accidentally got into to-do...", lines=3)
    
    generate_btn = gr.Button("Generate Meme", variant="primary")
    
    with gr.Row():
        output_image = gr.Image(label="Generated Meme")
        output_text = gr.Textbox(label="Generated Text", lines=4)
    
    # Явно отключаем генерацию внешнего API для этой кнопки через api_name=False
    generate_btn.click(
        fn=generate_demotivator,
        inputs=[product, pain],
        outputs=[output_image, output_text],
        api_name=False
    )

# Запуск на порту Hugging Face Spaces
demo.launch(server_name="0.0.0.0", server_port=7860)
