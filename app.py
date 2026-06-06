import streamlit as st
from inference import generate_memes
import time

st.set_page_config(page_title="Memetic AI", page_icon="🤖", layout="centered")

st.title("🤖 Memetic AI")
st.subheader("Собственная fine-tuned модель для генерации мемов")
st.caption("Capstone Project — Георгий Руденко")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    product = st.text_input("**Product / Brand**", value="Gym App")

with col2:
    pain = st.text_input("**Customer Pain**", value="no motivation to exercise")

num = st.slider("Number of memes", 1, 5, 3)

if st.button("🚀 Generate Memes", type="primary"):
    if product.strip() and pain.strip():
        with st.spinner("Model is thinking..."):
            start = time.time()
            memes = generate_memes(product, pain, num)
            end = time.time()
        
        st.success(f"Done in {end-start:.1f} seconds!")
        for i, meme in enumerate(memes, 1):
            st.markdown(f"**{i}.** {meme}")
    else:
        st.error("Please fill both fields!")

st.markdown("---")
st.caption("Model: Qwen2.5-3B fine-tuned | For HSE Capstone")