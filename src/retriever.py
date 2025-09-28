# retriever.py - wrapper around vectorstore with keyword fallback
import os, glob
from vectorstore import VectorStore

class Retriever:
    def __init__(self, kb_dir="kb", top_k=3):
        self.kb_dir = kb_dir
        self.top_k = top_k
        self.vs = VectorStore()
    def build_index_if_needed(self):
        if self.vs.index is None:
            docs = {}
            import os
            for path in glob.glob(os.path.join(self.kb_dir, "*.md")):
                name = os.path.basename(path)
                with open(path, "r", encoding="utf-8") as f:
                    docs[name] = f.read()
            self.vs.index_docs(docs)
    def retrieve(self, query):
        # ensure index exists
        try:
            self.build_index_if_needed()
            hits = self.vs.search(query, k=self.top_k)
            if hits:
                # convert L2 distances to similarity-like score by negative distance
                for h in hits:
                    h["sim"] = -h["score"]
                return hits
        except Exception as e:
            print("Vector search failed:", e)
        # fallback keyword search
        return self.keyword_search(query)
    def keyword_search(self, query):
        hits = []
        import glob
        import os
        for path in glob.glob(os.path.join(self.kb_dir, "*.md")):
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()
            if query.lower() in txt.lower():
                hits.append({"id": os.path.basename(path), "text": txt, "sim": 1.0})
        return hits[:self.top_k]
