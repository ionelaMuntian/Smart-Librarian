# app.py — Smart Librarian (LOCAL: Sentence-Transformers + GPT4All)
import os, json, re
from dataclasses import dataclass
from typing import List, Any
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

import chromadb
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All

console = Console()

DATA_PATH = os.path.join(os.path.dirname(__file__), "book_summaries.json")
DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db_local")
COLLECTION_NAME = "books_rag_local"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# GPT4All descarcă automat modelul la prima rulare (~3-4GB)
GPT4ALL_MODEL = "orca-mini-3b-gguf2-q4_0.gguf"

BAD_WORDS = {"idiot", "prost", "jignire", "ură", "urât", "hate", "fuck", "shit"}

@dataclass
class BookDoc:
    title: str
    short_summary: str
    full_summary: str
    themes: list

def load_books() -> List[BookDoc]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [BookDoc(**b) for b in data]

def get_or_create_collection(chroma_client) -> Any:
    try:
        return chroma_client.get_collection(COLLECTION_NAME)
    except Exception:
        return chroma_client.create_collection(COLLECTION_NAME)

def build_index(emb_model: SentenceTransformer, books: List[BookDoc]):
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = get_or_create_collection(chroma_client)

    try:
        if collection.count() >= len(books):
            return
    except Exception:
        pass

    docs, ids, metas = [], [], []
    for b in books:
        doc = f"Titlu: {b.title}\nTeme: {', '.join(b.themes)}\nRezumat scurt: {b.short_summary}"
        docs.append(doc)
        ids.append(b.title)
        # FIX: transformăm lista themes într-un șir de caractere
        metas.append({"title": b.title, "themes": ", ".join(b.themes)})

    vectors = emb_model.encode(docs, normalize_embeddings=True).tolist()
    collection.add(ids=ids, documents=docs, embeddings=vectors, metadatas=metas)

def query_similar(emb_model: SentenceTransformer, query: str, top_k: int = 3):
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = get_or_create_collection(chroma_client)
    q_emb = emb_model.encode([query], normalize_embeddings=True).tolist()
    return collection.query(query_embeddings=q_emb, n_results=top_k)

def local_get_summary_by_title(books: List[BookDoc], title: str) -> str:
    for b in books:
        if b.title.strip().lower() == title.strip().lower():
            return b.full_summary
    return "Nu am găsit această carte în baza de date locală."

def has_profanity(text: str) -> bool:
    t = text.lower()
    return any(bad in t for bad in BAD_WORDS)

def choose_with_llm(gpt: GPT4All, candidates: List[dict], user_q: str):
    """
    Îi cerem modelului să aleagă UN titlu dintre candidați și să răspundă JSON.
    Dacă nu reușește să producă JSON valid, facem fallback pe top-1.
    """
    cand_block = "\n".join([f"- {c['title']} :: {c['doc']}" for c in candidates])
    prompt = f"""
You are a helpful Romanian librarian bot. Alege exact UN titlu din lista de candidați care se potrivește cel mai bine întrebării.
Răspunde STRICT în format JSON cu cheile: "title" și "reply".
"reply" trebuie să fie un răspuns conversațional scurt (~120 cuvinte) în limba română.

Întrebare: {user_q}

Candidați:
{cand_block}

Exemplu răspuns:
{{"title": "1984", "reply": "Recomand '1984' pentru că ..."}}
"""
    out = gpt.generate(prompt, temp=0.2, max_tokens=300)
    # încearcă să extragi JSON
    m = re.search(r"\{.*\}", out, flags=re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict) and "title" in data and "reply" in data:
                return data["title"], data["reply"]
        except Exception:
            pass
    # fallback: top-1
    title = candidates[0]["title"] if candidates else "—"
    reply = f"Îți recomand {title} (cel mai relevant după căutarea semantică)."
    return title, reply

def main():
    console.print(Panel.fit("[bold green]Smart Librarian – LOCAL (RAG + GPT4All)[/bold green]"))
    books = load_books()
    emb_model = SentenceTransformer(EMBED_MODEL_NAME)
    build_index(emb_model, books)

    # încarcă modelul local (prima rulare îl descarcă automat)
    gpt = GPT4All(GPT4ALL_MODEL)

    console.print("Întrebare (ex: [i]Vreau o carte despre prietenie și magie[/i]) sau [bold]exit[/bold] pentru a ieși.\n")
    while True:
        user_q = Prompt.ask("[bold cyan]Tu[/bold cyan]").strip()
        if user_q.lower() in {"exit", "quit", "q"}:
            console.print("La revedere! 👋")
            break
        if has_profanity(user_q):
            console.print("[yellow]Hai să păstrăm conversația politicos. Întrebare nouă?[/yellow]")
            continue

        results = query_similar(emb_model, user_q, top_k=3)
        candidates = []
        for i in range(len(results.get("ids", [[]])[0])):
            candidates.append({
                "title": results["ids"][0][i],
                "doc": results["documents"][0][i],
                "meta": results["metadatas"][0][i]
            })

        if not candidates:
            console.print("[red]Nu am găsit candidați.[/red]")
            continue

        chosen_title, short_reply = choose_with_llm(gpt, candidates, user_q)
        full = local_get_summary_by_title(books, chosen_title)

        final = f"{short_reply}\n\n[Rezumat complet pentru {chosen_title}]\n{full}"
        console.print(Panel.fit(final))

if __name__ == "__main__":
    main()
