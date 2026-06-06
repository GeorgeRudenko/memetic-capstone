import streamlit as st
import torch
from unsloth import FastLanguageModel
from typing import List

st.set_page_config(page_title="Memetic AI", page_icon="🤖", layout="centered")

st.title("🤖 Memetic AI - Meme Generator")
st.markdown("### Fine-tuned Qwen2.5-3B for SMM Campaigns")
st.caption("Capstone Project — George Rudenko")

@st.cache_resource
def load_model():
    with st.spinner("Loading AI model... (first time takes ~2 minutes)"):
        model, tokenizer = FastLanguageModel.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    return model, tokenizer

def generate_memes(product: str, pain: str, num: int = 3) -> List[str]:
    model, tokenizer = load_model()
    
    prompt = f"""### Instruction:
Create exactly {num} short, funny and engaging meme texts for SMM.

### Product: {product}
### Pain: {pain}

### Response:"""
    
    inputs = tokenizer([prompt], return_tensors="pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.85,
        top_p=0.9,
        repetition_penalty=1.15,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    lines = [ln.strip() for ln in response.split("\n") if ln.strip()]
    memes = []
    for ln in lines:
        if len(ln) > 5:
            if ln[0].isdigit() and '.' in ln[:3]:
                ln = ln.split('.', 1)[1].strip()
            memes.append(ln)
    
    while len(memes) < num:
        memes.append(f"🔥 {product} fixes {pain} 🔥")
    
    return memes[:num]

col1, col2 = st.columns(2)

with col1:
    product = st.text_input("📦 Product / Brand", value="Gym App")

with col2:
    pain = st.text_input("😫 Customer Pain Point", value="no motivation to exercise")

num = st.slider("📝 Number of memes", 1, 5, 3)

if st.button("✨ Generate Memes", type="primary", use_container_width=True):
    if not product.strip() or not pain.strip():
        st.error("Please fill in both fields!")
    else:
        with st.spinner("🧠 Creating viral memes..."):
            try:
                memes = generate_memes(product.strip(), pain.strip(), num)
                st.success("✅ Memes generated!")
                st.divider()
                for i, meme in enumerate(memes, 1):
                    st.markdown(f"**{i}.** {meme}")
            except Exception as e:
                st.error(f"Error: {e}")

st.divider()
st.markdown("[GitHub Repository](https://github.com/GeorgeRudenko/memetic-capstone)")
