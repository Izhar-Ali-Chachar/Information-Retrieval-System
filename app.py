import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from src.helper import get_pdf_text_with_langchain, get_text_chunks, setup_chain

load_dotenv()

st.set_page_config(page_title="PDF Chat with Gemini", layout="centered")
st.title("📄 Chat with your PDF using Gemini")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Processing PDF..."):
        text = get_pdf_text_with_langchain(tmp_path)
        chunks = get_text_chunks(text)
        chain = setup_chain(chunks)
        st.success("PDF is ready! Ask your questions below.")

    os.remove(tmp_path)

    query = st.text_input("Ask a question about the PDF:")

    if query:
        with st.spinner("Generating answer..."):
            answer = chain.invoke(query)
            st.markdown("### 💬 Answer:")
            st.write(answer)
