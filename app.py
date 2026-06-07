import streamlit as st

st.set_page_config(page_title="Memetic - AI Meme Generator", layout="wide")

st.title("🤡 Memetic - AI-Powered Meme Generator")

st.write("Generate modern selling memes using a fine-tuned model.")

product = st.text_input("Product / Service", "Gym App")
pain = st.text_area("User Pain Point", "Users lack motivation to exercise regularly")

if st.button("Generate Memes"):
    st.success("Memes generated! (Demo mode - model will be loaded here)")
    st.write("1. When you finally open the app and suddenly feel motivated...")
    st.write("2. Motivation? I don't know her...")
    st.write("3. Opened Gym App and instantly wanted to live again")

st.caption("Fine-tuned model for HSE Capstone Project")