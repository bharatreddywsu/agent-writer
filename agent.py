import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA

# ✅ Set Hugging Face API token (must be set in your environment)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ✅ Load documents from the 'docs' folder
loader = DirectoryLoader('docs')
documents = loader.load()

# ✅ Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

# ✅ Create embeddings using HuggingFace sentence transformer
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Create FAISS vector store
db = FAISS.from_documents(splits, embeddings)

# ✅ Set up retriever
retriever = db.as_retriever()

# ✅ Set up the LLM from Hugging Face (using bloomz-560m)
llm = HuggingFaceEndpoint(
    repo_id="bigscience/bloomz-560m",
    temperature=0.5,
)

# ✅ Build Retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

print("✅ AI Research Assistant is ready! Ask questions or type 'exit' to quit.\n")

# ✅ Interactive Q&A loop
while True:
    query = input("Your question: ")
    if query.lower() == 'exit':
        break
    try:
        answer = qa_chain.invoke({"query": query})
        print("\nAnswer:", answer, "\n")
    except Exception as e:
        print("\n⚠️ Error:", e, "\n")
