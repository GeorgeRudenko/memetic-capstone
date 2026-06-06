# Memetic AI — AI-Powered Meme Generator

**Capstone Project**

Собственная fine-tuned модель (Qwen2.5-3B) для генерации мемов по Product + Pain Point.

## Как запустить

### Веб-приложение (Streamlit)
```bash
git clone https://github.com/GeorgeRudenko/memetic-capstone.git
cd memetic-capstone
pip install -r requirements.txt
streamlit run app.py
```

### Или через скрипт
```bash
python inference.py
```

## Структура проекта
- `app.py` — Streamlit веб-интерфейс
- `inference.py` — генерация мемов
- `data_processing.py` — подготовка данных
- `fine_tuning/train.py` — скрипт обучения
- `Memetic_MVP_v2.pptx` — презентация

**Автор:** Георгий Руденко