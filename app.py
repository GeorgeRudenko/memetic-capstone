import gradio as gr
from meme_generator_local import MemeGenerator

# Инициализация генератора
generator = MemeGenerator()
generator.load_templates("meme_templates_clean.json")

def generate_demotivator(product, pain):
    if not product or not pain:
        return None, "Пожалуйста, заполните оба поля"
    
    output_path = "generated_meme.png"
    text = generator.generate_meme(product.strip(), pain.strip(), output_path)
    return output_path, text

# Интерфейс
with gr.Blocks(title="Memetic Generator") as demo:
    gr.Markdown("# 🧠 Memetic Generator")
    gr.Markdown("Генератор демотиваторов на основе fine-tuned Qwen2.5-7B + LoRA")
    
    with gr.Row():
        product = gr.Textbox(label="Продукт / Ситуация", placeholder="Jira, Slack, Notion...")
        pain = gr.Textbox(label="Боль / Проблема", placeholder="backlog task accidentally got into to-do...", lines=3)
    
    generate_btn = gr.Button("Сгенерировать демотиватор", variant="primary")
    
    with gr.Row():
        output_image = gr.Image(label="Сгенерированный демотиватор")
        output_text = gr.Textbox(label="Сгенерированный текст", lines=4)
    
    generate_btn.click(
        fn=generate_demotivator,
        inputs=[product, pain],
        outputs=[output_image, output_text]
    )

demo.launch()
