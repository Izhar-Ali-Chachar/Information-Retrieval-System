from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS

def get_pdf_text_with_langchain(pdf_path):
    all_text = ""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()  # returns a list of Document objects
    for doc in documents:
        all_text += doc.page_content + "\n"
    return all_text

result = get_pdf_text_with_langchain('check.pdf')

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    return text_splitter.create_documents([text])



def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    return FAISS.from_documents(chunks, embeddings)

chunks = get_text_chunks(result)

print(get_vector_store(chunks))
