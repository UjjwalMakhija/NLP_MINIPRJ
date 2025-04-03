import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import torch

model_path = "Ujjwal0404/English_French_Tranlation"  
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

st.title("Neural Machine Translation (NMT)")
st.markdown("Translate English text into another language using a fine-tuned MarianMT model.")

text = st.text_area("Enter text in English:", "")

if st.button("Translate to French"):
    if text.strip():
     
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            translated = model.generate(**inputs)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)

        st.subheader("Translated Text in French:")
        st.success(result)
    else:
        st.warning("Please enter some text to translate.")
