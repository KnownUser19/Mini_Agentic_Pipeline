# tests.py - run multiple queries and record latencies + notes
import time, json
from src.main import handle_query
from src.retriever import Retriever
from src.reasoner import Reasoner
from src.actor_csv import CSVActor

retriever = Retriever(kb_dir="kb", top_k=3)
reasoner = Reasoner(version="v2")
actor = CSVActor(csv_path="prices.csv")

QUERIES = [
 "What's the price of Coffee COF123?",
 "Do you have Pen PEN456 and what's the price?",
 "Explain cloud VM pricing basics.",
 "I need taxes applied to a $100 price with 5% discount",
 "Show top 3 stationery items",
 "Is there a laptop under $1000?",
 "What is a token bucket rate-limiting?",
 "Give me product details for SKU LAP789"
]

results = []
for q in QUERIES:
    t0 = time.perf_counter()
    final, trace = handle_query(q, retriever, reasoner, actor)
    t1 = time.perf_counter()
    results.append({
       "query": q,
       "final": final,
       "trace": trace,
       "total_latency_s": t1-t0
    })
    print(f"Ran: {q} -> {results[-1]['total_latency_s']:.3f}s")

with open("test_results.json","w",encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Saved test_results.json")
