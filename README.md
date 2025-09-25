# Bible-Agent
Let's set up a sample project using LangChain + Bible data. This project will be a simple AI agent that can answer questions about the Bible based on a local dataset. Later, you can add Bach as a second domain.

author: Beni Dhomi 
keywords: langchain, bible, openai, chromadb, vector-database, rag

## What are we building?
A Python project that:
1. Loads Bible texts (e.g., KJV, ALB or NBV)
2. Converts these texts into a searchable **vector database**
3. Uses an AI agent (via OpenAI or a local model) to answer questions using LangChain + RAG

---

## Prerequisites
You need:
- Python 3.10+
- An local model via Ollama
---

## Project Structure
```
bible-agent/
├── data/
│   │   bible1.txt  # Complete text of the Bible
│   └── bible2.txt  # Complete text of the Bible
├── main.py         # Main script
├── .env            # API keys
```

---

## Step 1: Preparing Bible Data
You can use a text file with the Bible, for example:
```txt
Genesis 1:1 In the beginning, God created the heaven and the earth.
Genesis 1:2 And the earth was without form, and void...
...
```

## 
brew install ollama 
ollama pull minstral
ollama run minstral

other terminal: python main.py

## What can you do now?
- Ask questions like: "What does the Bible say about love?"
- Later expand with Bach-data (MIDI, texts, biographies)
- Build an interface with Streamlit or Gradio

- trick to uninstall all venv packages: 
```pip freeze | xargs pip uninstall -y```
- reinstall: 
```pip install -r requirements.txt```


# UI
```python ui.py``` and then http://127.0.0.1:5000/