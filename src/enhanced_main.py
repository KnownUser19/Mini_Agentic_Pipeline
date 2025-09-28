# src/enhancedmain.py
# Enhanced orchestrator with shared context and robust CSV normalization

import os
import time
import json
import sys
import re
from dotenv import load_dotenv

load_dotenv()

# Ensure we can import sibling modules in src/
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SRC_DIR)

from retriever import Retriever
from reasoner import Reasoner
from actor_csv import CSVActor
from web_search_actor import WebSearchActor
from api_actor import APIActor
from utils import make_trace_entry, pretty_print_trace


class AgenticPipeline:
    def __init__(self, kb_dir="kb", csv_path="prices.csv"):
        # Initialize components
        self.retriever = Retriever(kb_dir=kb_dir, top_k=int(os.getenv("TOP_K", 3)))
        self.reasoner = Reasoner(version=os.getenv("REASONER_VERSION", "v2"))
        self.csv_actor = CSVActor(csv_path=csv_path)
        self.web_actor = WebSearchActor()
        self.api_actor = APIActor()

        # Shared context/state
        self.shared_context = {
            "query_history": [],
            "tool_usage_count": {"csv": 0, "web": 0, "api": 0},
            "total_latency": 0.0,
            "session_id": int(time.time())
        }

    def _extract_sku(self, q: str) -> str | None:
        """
        Extract SKU tokens robustly (e.g., 'find sku PEN456.' -> 'PEN456')
        - case-insensitive
        - tolerates separators like 'sku:', 'sku-', 'sku '
        - strips trailing punctuation
        """
        m = re.search(r"\bsku[-\s:]*([a-z0-9\-]+)\b", q.lower(), flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        return None

    def _extract_product(self, q: str) -> str:
        """
        Extract a normalized product term from natural queries:
        'what is price of coffee?' -> 'coffee'
        Fallback: cleaned lowercased string without punctuation.
        """
        ql = q.lower()
        m = re.search(r"price of\s+([a-z0-9\-\s]+)", ql)
        term = m.group(1) if m else ql
        # Strip punctuation and extra spaces
        term = re.sub(r"[^a-z0-9\s\-]", "", term).strip()
        return term

    def _csv_lookup_with_retries(self, query: str):
        """
        Perform CSV lookups with multiple normalized candidates in order:
        1) original query
        2) punctuation-stripped, lowercased query
        3) extracted SKU token (uppercase)
        4) extracted product term
        Returns: (result, total_latency, attempts_log)
        """
        attempts = []
        candidates = []

        # 1) raw query
        candidates.append(("raw", query))

        # 2) cleaned lowercase (punctuation-stripped)
        cleaned = re.sub(r"[^a-z0-9\s\-]", "", query.lower()).strip()
        if cleaned and cleaned != query:
            candidates.append(("cleaned", cleaned))

        # 3) extracted SKU
        sku = self._extract_sku(query)
        if sku:
            candidates.append(("sku", sku))

        # 4) product term
        prod = self._extract_product(query)
        if prod and prod not in {query, cleaned, sku}:
            candidates.append(("product", prod))

        total_latency = 0.0
        for label, cand in candidates:
            t0 = time.perf_counter()
            res, latency = self.csv_actor.lookup(cand)
            t1 = time.perf_counter()
            elapsed = latency if latency is not None else (t1 - t0)
            total_latency += elapsed
            attempts.append({"attempt": label, "query": cand, "latency_s": elapsed, "rows": len(res) if isinstance(res, list) else (1 if res else 0)})
            if isinstance(res, list) and len(res) > 0:
                return res, total_latency, attempts

        # No results
        return [], total_latency, attempts

    def handle_query(self, query: str) -> tuple:
        """Enhanced query handling with multiple tools and shared context"""
        trace = []

        # Add to query history
        self.shared_context["query_history"].append({
            "query": query,
            "timestamp": time.time()
        })

        # 1) Retrieve from knowledge base
        t0 = time.perf_counter()
        kb_hits = self.retriever.retrieve(query)
        t1 = time.perf_counter()
        retrieval_latency = t1 - t0
        self.shared_context["total_latency"] += retrieval_latency

        trace.append(make_trace_entry("retrieval", {
            "query": query,
            "hits": kb_hits,
            "latency_s": retrieval_latency,
            "context": {"query_count": len(self.shared_context["query_history"])}
        }))

        # 2) Reasoner decides next action
        t0 = time.perf_counter()
        decision = self.reasoner.decide(query, kb_hits, self.shared_context)
        t1 = time.perf_counter()
        reasoning_latency = t1 - t0
        self.shared_context["total_latency"] += reasoning_latency

        trace.append(make_trace_entry("reasoner_decision", {
            "decision": decision,
            "latency_s": reasoning_latency,
            "context": {"total_tool_calls": sum(self.shared_context["tool_usage_count"].values())}
        }))

        # 3) Execute tool action
        tool_result = None
        tool_latency = None
        tool_meta = {}

        if decision.get("decision") == "TOOL_CSV":
            tool_query = decision.get("tool_args", {}).get("query", query)
            # multi-attempt CSV lookup with normalization
            res, total_latency, attempts = self._csv_lookup_with_retries(tool_query)
            tool_result = res
            tool_latency = total_latency
            tool_meta["attempts"] = attempts
            self.shared_context["tool_usage_count"]["csv"] += 1

        elif decision.get("decision") == "TOOL_WEB":
            tool_query = decision.get("tool_args", {}).get("query", query)
            t0 = time.perf_counter()
            tool_result, tool_latency = self.web_actor.search(tool_query)
            t1 = time.perf_counter()
            if tool_latency is None:
                tool_latency = t1 - t0
            self.shared_context["tool_usage_count"]["web"] += 1

        elif decision.get("decision") == "TOOL_API":
            api_endpoint = decision.get("tool_args", {}).get("endpoint", "posts")
            api_method = decision.get("tool_args", {}).get("method", "GET")
            api_params = decision.get("tool_args", {}).get("params", {})
            t0 = time.perf_counter()
            tool_result, tool_latency = self.api_actor.call_api(api_endpoint, api_method, api_params)
            t1 = time.perf_counter()
            if tool_latency is None:
                tool_latency = t1 - t0
            self.shared_context["tool_usage_count"]["api"] += 1

        if tool_result is not None:
            self.shared_context["total_latency"] += (tool_latency or 0.0)
            trace.append(make_trace_entry("tool_call", {
                "tool": decision.get("decision", "unknown"),
                "tool_query": decision.get("tool_args", {}),
                "result": tool_result,
                "latency_s": tool_latency,
                "meta": tool_meta,
                "context": {"tool_usage": self.shared_context["tool_usage_count"]}
            }))

        # 4) Generate final answer
        # Record decision so Reasoner.final_answer can enforce provenance rules
        self.shared_context["last_decision"] = decision

        t0 = time.perf_counter()
        # Reasoner.final_answer accepts shared_context as an optional 4th param
        final = self.reasoner.final_answer(query, kb_hits, tool_result, self.shared_context)
        t1 = time.perf_counter()
        final_latency = t1 - t0
        self.shared_context["total_latency"] += final_latency

        trace.append(make_trace_entry("final_answer", {
            "final": final,
            "latency_s": final_latency,
            "context": {"total_session_latency": self.shared_context["total_latency"]}
        }))

        return final, trace

    def get_session_stats(self) -> dict:
        """Get session statistics"""
        return {
            "total_queries": len(self.shared_context["query_history"]),
            "tool_usage": self.shared_context["tool_usage_count"],
            "total_latency": self.shared_context["total_latency"],
            "session_id": self.shared_context["session_id"]
        }


if __name__ == "__main__":
    # Check for environment configuration
    print("🚀 Starting Enhanced Mini Agentic Pipeline...")
    print("Checking environment configuration...\n")

    if not os.path.exists(".env"):
        print("ℹ️  No .env file found. Using default settings.")
        print("💡 Local sentence-transformers embeddings are free and don't require API keys!")
        print("📝 Optional: Create .env file to customize settings (see ENV_CONFIG.md)\n")

    openai_key = os.getenv("OPENAI_API_KEY")
    hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
    use_hf_embeddings = os.getenv("USE_HF_EMBEDDINGS", "true").lower() == "true"

    if use_hf_embeddings:
        print("✅ Using Hugging Face embeddings (free local models)")
        if not openai_key:
            print("ℹ️  OPENAI_API_KEY not set. Will use HF embeddings only.")
    else:
        if not openai_key:
            print("⚠️  Warning: OPENAI_API_KEY not set and HF embeddings disabled.")
            print("Please set either OPENAI_API_KEY or enable HF embeddings (USE_HF_EMBEDDINGS=true)")

    if not hf_token:
        print("ℹ️  HUGGINGFACE_API_TOKEN not set. Local HF models will work fine!")

    print("Environment check complete.\n")

    # Initialize enhanced pipeline
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pipeline = AgenticPipeline(
        kb_dir=os.path.join(project_root, "kb"),
        csv_path=os.path.join(project_root, "prices.csv")
    )

    print("Enhanced Mini Agentic Pipeline ready!")
    print("Available tools: CSV lookup, Web search, REST API calls")
    print("Type queries (or 'exit', 'stats'):")

    while True:
        q = input("> ").strip()
        if q in ("exit", "quit"):
            break
        elif q == "stats":
            stats = pipeline.get_session_stats()
            print(f"\n=== SESSION STATISTICS ===")
            print(f"Total queries: {stats['total_queries']}")
            print(f"Tool usage: {stats['tool_usage']}")
            print(f"Total latency: {stats['total_latency']:.3f}s")
            print(f"Session ID: {stats['session_id']}\n")
            continue

        try:
            final, trace = pipeline.handle_query(q)
            print("\n=== FINAL ANSWER ===")
            print(final)
            print("\n=== TRACE ===")
            print(pretty_print_trace(trace))
            print("\n")
        except Exception as e:
            print(f"Error: {e}")
            print("Please check your configuration and try again\n")
