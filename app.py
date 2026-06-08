import streamlit as st
from meme_generator_local import MemeGenerator

st.set_page_config(page_title="Memetic Generator", page_icon="🧠", layout="centered")

st.title("🧠 Memetic Generator")
st.markdown("Генератор демотиваторов")

@st.cache_resource
def get_generator():
    gen = MemeGenerator()
    gen.load_templates("meme_templates_clean.json")
    return gen

generator = get_generator()

product = st.text_input("Продукт / Ситуация", value="Jira")
pain = st.text_area("Боль / Проблема", value="backlog task accidentally got into to-do", height=100)

if st.button("Сгенерировать демотиватор", type="primary"):
    if product and pain:
        with st.spinner("Генерирую..."):
            output_path = "generated_meme.png"
            generator.generate_meme(product.strip(), pain.strip(), output_path)
            st.image(output_path, use_container_width=True)
    else:
        st.warning("Заполни оба поля")
