# 🧠 Memetic Generator

An automated pipeline that generates contextual **demotivator-style memes** using a fine-tuned **Qwen2.5-7B** model with LoRA. The system combines semantic template retrieval (Sentence Transformers) with high-quality text generation and custom Pillow-based rendering.

## 🚀 Live Demo

**Working Production Version:** [https://huggingface.co/spaces/georgiyruden/memetic-generator](https://huggingface.co/spaces/georgiyruden/memetic-generator)

---

## 🛠️ Technology Stack

- **LLM**: Qwen2.5-7B-Instruct fine-tuned with LoRA
- **Retrieval**: `sentence-transformers` (semantic search over meme templates)
- **UI**: Gradio
- **Image Rendering**: Pillow (PIL) — classic demotivator style (black background + white border + dual text)
- **Model Storage**: Git LFS

## 🔄 Pipeline

1. User provides **Product** + **Pain** (e.g. "Jira" + "backlog task accidentally got into to-do")
2. System retrieves the most relevant meme template using semantic embeddings
3. Fine-tuned LLM generates sharp, contextual caption (two lines)
4. Text is rendered on the template using Pillow in classic demotivator format

## 🚀 How to Run Locally

```bash
git clone https://github.com/GeorgeRudenko/memetic-capstone.git
cd memetic-capstone

pip install -r requirements.txt
python app.py
```

## 📁 Project Structure

```
memetic-capstone/
├── app.py
├── meme_generator_local.py
├── meme_templates_clean.json
├── development_notebook.ipynb
├── checkpoints/qwen_lora_memes_v1/   # LoRA weights
├── requirements.txt
└── README.md
```

## 🔗 Important Notes

- The model file (`adapter_model.safetensors`) is stored using **Git LFS**.
- For best results when cloning: `git lfs install` before cloning.
- The Gradio interface is also available via Google Colab notebook for quick testing.

## 📄 License

MIT License