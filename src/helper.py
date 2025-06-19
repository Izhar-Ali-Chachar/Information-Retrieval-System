from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv


load_dotenv()

def get_pdf_text_with_langchain(pdf_path):
    all_text = ""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()  # returns a list of Document objects
    for doc in documents:
        all_text += doc.page_content + "\n"
    return all_text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    return text_splitter.create_documents([text])



def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    return FAISS.from_documents(chunks, embeddings)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def setup_chain(docs):
    vector_store = get_vector_store(docs)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Use ONLY the following context to answer the question. 
If the context is insufficient, respond with: "I don't know."

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)


    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-05-20",
    )

    parser = StrOutputParser()

    return (
            RunnableParallel({
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }) | prompt | model | parser
        )