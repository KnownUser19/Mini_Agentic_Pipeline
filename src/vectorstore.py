import os
import json
import numpy as np
import faiss
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Environment variables
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
USE_HF_EMBEDDINGS = os.getenv("USE_HF_EMBEDDINGS", "true").lower() == "true"  # Default to HF embeddings

INDEX_DIR = Path("data")
INDEX_DIR.mkdir(exist_ok=True)

# Initialize OpenAI client (only if not using HF embeddings)
client = None
if not USE_HF_EMBEDDINGS:
    try:
        from openai import OpenAI
        client = OpenAI()
    except ImportError:
        print("Warning: openai not installed. OpenAI embeddings fallback will not work.")

# Initialize Hugging Face sentence-transformers (default, no token needed for local models)
hf_model = None
if USE_HF_EMBEDDINGS:
    try:
        from sentence_transformers import SentenceTransformer
        print(f"Loading Hugging Face embedding model: {HF_EMBEDDING_MODEL}")
        print("Note: Local sentence-transformers models are free and don't require API tokens!")
        hf_model = SentenceTransformer(HF_EMBEDDING_MODEL)
        print(f"✅ Successfully loaded Hugging Face embedding model: {HF_EMBEDDING_MODEL}")
    except ImportError:
        print("❌ Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")
        print("Falling back to OpenAI embeddings (requires API key).")
        USE_HF_EMBEDDINGS = False
    except Exception as e:
        print(f"❌ Warning: Failed to load HF embedding model: {e}")
        print("Falling back to OpenAI embeddings (requires API key).")
        USE_HF_EMBEDDINGS = False

class VectorStore:
    def __init__(self, index_path="data/index.faiss", meta_path="data/meta.json"):
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.meta = []
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self._load()

    def _embed(self, texts):
        """Return embeddings as NumPy array (float32)."""
        if USE_HF_EMBEDDINGS and hf_model is not None:
            embeddings = hf_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings.astype('float32')
        else:
            if client is None:
                raise RuntimeError("Neither HF embeddings nor OpenAI client available. Check your configuration.")
            resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
            return np.array([r.embedding for r in resp.data], dtype='float32')

    def index_docs(self, docs: dict):
        """
        docs: dict[id] = text
        Builds FAISS index and stores metadata.
        Uses cosine similarity (FAISS IndexFlatIP) for embeddings.
        """
        ids = []
        texts = []
        for doc_id, text in docs.items():
            ids.append(doc_id)
            texts.append(text)

        embeds = self._embed(texts)
        # normalize for cosine similarity
        embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)

        dim = embeds.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity
        self.index.add(embeds)

        self.meta = [{"id": ids[i], "text": texts[i]} for i in range(len(ids))]

        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)

        print(f"Indexed {len(ids)} documents with dimension {dim}.")

    def _load(self):
        """Load FAISS index and metadata."""
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        print(f"Loaded index with {len(self.meta)} documents.")

    def search(self, query, k=3):
        """Return top-k hits for a query."""
        if self.index is None:
            return []

        q_emb = self._embed([query])
        # normalize for cosine similarity
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

        D, I = self.index.search(q_emb, k)
        hits = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:  # skip empty results
                continue
            meta = self.meta[idx]
            hits.append({"id": meta["id"], "text": meta["text"], "score": float(dist)})
        return hits
