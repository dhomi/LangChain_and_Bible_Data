import os
import re
import subprocess

DATA_FOLDER = 'data'
MODEL_NAME = 'mistral'  # Ollama model name

def load_bible_texts(data_folder):
    """Load all txt files from data folder into a dict: {book_name: [passages]}"""
    bible = {}
    for filename in os.listdir(data_folder):
        if filename.endswith('.txt'):
            path = os.path.join(data_folder, filename)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Parse by book header ### BookName
                books = re.split(r'^###\s*(.+)$', content, flags=re.MULTILINE)
                # The split creates list: ['', bookname1, text1, bookname2, text2, ...]
                # So we iterate pairs (book, text)
                for i in range(1, len(books), 2):
                    book = books[i].strip()
                    text = books[i+1].strip()
                    passages = text.split('\n')
                    if book not in bible:
                        bible[book] = []
                    # Filter out empty lines
                    bible[book].extend([p.strip() for p in passages if p.strip()])
    return bible

def search_passages(bible, query, max_passages=10):
    """Basic search: return passages containing any keyword from query."""
    results = []
    query_words = set(query.lower().split())
    for book, passages in bible.items():
        for p in passages:
            # Check if any query word appears in passage (case-insensitive)
            words_in_p = set(p.lower())
            if any(word in p.lower() for word in query_words):
                results.append(f"### {book}\n{p}")
                if len(results) >= max_passages:
                    return results
    return results

def build_prompt(passages, question):
    """Build prompt with retrieved passages and question."""
    passages_text = "\n\n".join(passages)
    prompt = (
        f"Based only on the following Bible passages:\n\n"
        f"{passages_text}\n\n"
        f"Answer the question: \"{question}\"\n"
        f"If the answer is not in the passages, say you don't know."
    )
    return prompt

def ask_ollama(prompt):
    """Call ollama CLI with prompt and return model output.

    The Ollama CLI doesn't accept a `--prompt` flag — pass the prompt via stdin.
    """
    try:
        process = subprocess.run(
            ['ollama', 'run', MODEL_NAME],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60  # seconds; adjust as needed
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Ollama CLI timed out (increase timeout or use a smaller prompt)")

    if process.returncode != 0:
        # Include stderr and stdout for debugging
        raise RuntimeError(
            f"Ollama CLI error (rc={process.returncode}). stderr:\n{process.stderr.strip()}\nstdout:\n{process.stdout.strip()}"
        )

    return process.stdout.strip()

def main():
    print("Loading Bible texts...")
    bible = load_bible_texts(DATA_FOLDER)
    print(f"Loaded books: {list(bible.keys())}")

    while True:
        question = input("\nEnter your question (or 'exit' to quit): ").strip()
        if question.lower() == 'exit':
            break

        print("Searching relevant passages...")
        passages = search_passages(bible, question, max_passages=5)

        if not passages:
            print("No relevant Bible passages found.")
            continue

        prompt = build_prompt(passages, question)
        print("\nAsking Ollama model...")
        answer = ask_ollama(prompt)
        print("\nAnswer:")
        print(answer)

if __name__ == '__main__':
    main()