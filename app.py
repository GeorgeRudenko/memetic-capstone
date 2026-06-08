import gradio as gr
from meme_generator_local import MemeGenerator

# Инициализация генератора
generator = MemeGenerator()
generator.load_templates("meme_templates_clean.json")

def generate(product, pain):
    if not product or not pain:
        return None, "Пожалуйста, заполните оба поля"
    
    output_path = "generated_meme.png"
    text = generator.generate_meme(product.strip(), pain.strip(), output_path)
    return output_path, text

# Примеры
examples = [
    ["Jira", "backlog task accidentally got into to-do"],
    ["Slack", "constant notifications and hundreds of unread channels"],
    ["Notion", "I created 47 databases and can't find anything"],
    ["Zoom", "another 3-hour meeting that could have been an email"],
    ["GitHub", "I forgot to make a pull request again"]
]

with gr.Blocks(title="Memetic Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Memetic Generator")
    gr.Markdown("### Генератор демотиваторов на базе fine-tuned Qwen2.5-7B")
    
    with gr.Row():
        with gr.Column(scale=1):
            product = gr.Textbox(label="Продукт / Ситуация", placeholder="Jira, Slack, Notion...", value="Jira")
            pain = gr.Textbox(label="Боль / Проблема", placeholder="backlog task accidentally got into to-do...", lines=3, value="backlog task accidentally got into to-do")
            
            generate_btn = gr.Button("Сгенерировать демотиватор", variant="primary", size="large")
        
        with gr.Column(scale=1):
            output_image = gr.Image(label="Результат", height=500)
            output_text = gr.Textbox(label="Сгенерированный текст", lines=3)
    
    gr.Examples(
        examples=examples,
        inputs=[product, pain],
        label="Примеры"
    )

    generate_btn.click(
        fn=generate,
        inputs=[product, pain],
        outputs=[output_image, output_text]
    )

demo.launch()