import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
# Use the new first-party HuggingFace embeddings package
from langchain_huggingface import HuggingFaceEmbeddings
# Use the new Ollama package LLM
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Load all Bible text files from the data directory
data_dir = "data"
if not os.path.isdir(data_dir):
    raise SystemExit(f"Data directory not found: {data_dir}")

documents = []
for filename in os.listdir(data_dir):
    if filename.endswith(".txt"):
        path = os.path.join(data_dir, filename)
        loader = TextLoader(path, autodetect_encoding=True)
        try:
            documents.extend(loader.load())
        except Exception as exc:
            print(f"Warning: failed to load {path}: {exc}")

if not documents:
    raise SystemExit("No documents loaded from data directory.")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n[", "\n", " "]
)
texts = text_splitter.split_documents(documents)

# Create local embeddings using the new package
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store in Chroma vector database
# Optional: pass persist_directory="chroma_db" to persist the DB across runs
db = Chroma.from_documents(texts, embeddings)

# Create a retriever and QA chain using OllamaLLM (mistral)
retriever = db.as_retriever()
llm = OllamaLLM(model="mistral")  # requires langchain-ollama package and a running Ollama instance
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Example query
query = "What do the three Bible versions say about Genesis 1:1?"

# Use invoke (replacement for deprecated .run). invoke usually accepts a dict of inputs.
result = qa_chain.invoke({"query": query})

# Normalize result to a string to print
answer = None
if isinstance(result, dict):
    for key in ("result", "answer", "output_text", "text"):
        if key in result and result[key]:
            answer = result[key]
            break
    if answer is None and result:
        for v in result.values():
            if isinstance(v, str) and v.strip():
                answer = v
                break
elif isinstance(result, str):
    answer = result
else:
    answer = str(result)

print(answer)