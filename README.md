# 🧠 Memetic Generator

An automated local pipeline and web application that generates contextual demotivator memes. The project leverages a fine-tuned **Qwen2.5-7B** Large Language Model with tailored **LoRA** checkpoints to generate witty captions, combined with a **SentenceTransformer** retriever to select the most relevant visual templates.

## 🛠️ Technology Stack

*   **LLM Core:** Qwen2.5-7B (Fine-tuned using LoRA adapters)
*   **Vector Retrieval:** `sentence-transformers` for semantic matching of meme templates
*   **UI Framework:** Gradio 5 (Optimized with isolated client-side rendering)
*   **Deployment:** Configured for dynamic CPU/Float32 fallback environments
*   **Image Processing:** Pillow (PIL) for high-fidelity demotivator canvas generation

## 📍 Architecture & Pipeline

1.  **Semantic Retrieval:** The user inputs a product (e.g., Jira, Slack) and a specific pain point. The system uses embeddings to query a curated dataset (`meme_templates_clean.json`) and retrieve the best-fitting background meme template.
2.  **Contextual Generation:** The fine-tuned LLM receives an engineered prompt containing the user inputs and generates the upper and lower text for the demotivator, maintaining high contextual relevance.
3.  **Canvas Rendering:** The text and image are dynamically composited into a classic demotivator template layout.

## 🚀 Local Setup & Installation

### Prerequisites
*   Python 3.10+
*   Highly recommended: Virtual environment manager (`venv` or `conda`)

### 1. Clone the Repository
```bash
git clone https://github.com/GeorgeRudenko/memetic-capstone.git
cd memetic-capstone
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application Locally
```bash
python app.py
```

Once initialized, the interface will be served locally at http://localhost:7860.

## 📦 Deployment

The project is designed to run via Google Colab with Gradio's public link sharing for quick demonstrations. A stable deployment on platforms like Hugging Face Spaces was attempted but faced challenges with PyTorch installation in restricted CPU environments.

For the most reliable demo experience, use the provided Colab notebook.

## 📄 License

This project is open-source and available under the MIT License.