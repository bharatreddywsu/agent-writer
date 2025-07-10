import os
import sys
import importlib

# ── Patch to fix sqlite3 version issue ──
importlib.import_module("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# ── Langchain and HuggingFace Imports ──
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA

# ── Load Hugging Face API key from env ──
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ── Load and Split Documents ──
loader = DirectoryLoader('docs')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

# ── Embeddings and Vectorstore ──
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(splits, embeddings)

# ── Create Retriever and LLM ──
retriever = db.as_retriever()

llm = HuggingFaceEndpoint(
    repo_id="bigscience/bloomz-560m",
    temperature=0.5,
)

# ── Retrieval-based QA Chain ──
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

print("✅ AI Research Assistant is ready! Ask questions or type 'exit' to quit.\n")

# ── Interactive Loop ──
while True:
    query = input("Your question: ")
    if query.lower() == 'exit':
        break
    try:
        answer = qa_chain.invoke({"query": query})
        print("\nAnswer:", answer, "\n")
    except Exception as e:
        print("\n⚠️ Error:", e, "\n")
