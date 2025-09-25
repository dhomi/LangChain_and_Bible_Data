import os
import re
import subprocess
from flask import Flask, request, render_template_string

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

DATA_FOLDER = 'data'
MODEL_NAME = 'mistral'  # ollama model name

# --- Load & index Bible texts -----------------------------------------------

def load_bible_texts(data_folder):
    bible = {}
    for filename in os.listdir(data_folder):
        if filename.endswith('.txt'):
            path = os.path.join(data_folder, filename)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                books = re.split(r'^###\s*(.+)$', content, flags=re.MULTILINE)
                for i in range(1, len(books), 2):
                    book = books[i].strip()
                    text = books[i+1].strip()
                    passages = text.split('\n')
                    if book not in bible:
                        bible[book] = []
                    bible[book].extend([p.strip() for p in passages if p.strip()])
    return bible

def build_embedding_index(bible):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    corpus = []
    passage_refs = []
    for book, passages in bible.items():
        for p in passages:
            corpus.append(p)
            passage_refs.append(f"### {book}\n{p}")
    embeddings = model.encode(corpus, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity with normalized vectors
    index.add(embeddings)
    return index, passage_refs, model

def search_passages_vector(index, passage_refs, model, query, top_k=5):
    query_embedding = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding, top_k)
    results = [passage_refs[i] for i in indices[0] if i < len(passage_refs)]
    return results

# --- Build prompt, call Ollama CLI ------------------------------------------

def build_prompt(passages, question):
    passages_text = "\n\n".join(passages)
    prompt = (
        f"Based only on the following Bible passages:\n\n"
        f"{passages_text}\n\n"
        f"Answer the question: \"{question}\"\n"
        f"If the answer is not in the passages, say you don't know."
    )
    return prompt

def ask_ollama(prompt):
    process = subprocess.run(
        ['ollama', 'run', MODEL_NAME],
        input=prompt,
        capture_output=True,
        text=True
    )
    if process.returncode != 0:
        return f"Ollama CLI error: {process.stderr.strip()}"
    return process.stdout.strip()
# --- Flask Web app -----------------------------------------------------------

app = Flask(__name__)

HTML_TEMPLATE = '''
<html>
  <head><title>Bible Q&A with Ollama</title></head>
  <body>
    <h1>Bible Q&A with Ollama</h1>
    <form method="post">
      <textarea name="question" rows="3" cols="60">{{ question or '' }}</textarea><br><br>
      <input type="submit" value="Ask">
    </form>
    {% if answer %}
      <h2>Answer:</h2>
      <pre>{{ answer }}</pre>
    {% endif %}
  </body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    question = ''
    if request.method == 'POST':
        question = request.form['question'].strip()
        if question:
            passages = search_passages_vector(index, passage_refs, model, question, top_k=5)
            if passages:
                prompt = build_prompt(passages, question)
                answer = ask_ollama(prompt)
            else:
                answer = "No relevant Bible passages found."
    return render_template_string(HTML_TEMPLATE, answer=answer, question=question)

# --- Main --------------------------------------------------------------------

if __name__ == '__main__':
    print("Loading Bible texts...")
    bible = load_bible_texts(DATA_FOLDER)
    print(f"Loaded books: {list(bible.keys())}")

    print("Building passage embeddings index...")
    index, passage_refs, model = build_embedding_index(bible)

    print("Starting Flask web server on http://localhost:5000")
    app.run(debug=True)