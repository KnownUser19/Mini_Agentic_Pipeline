# src/reasoner.py
import os
import json
import requests
from dotenv import load_dotenv


load_dotenv()


# Environment variables
MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-large")
USE_HF_MODEL = os.getenv("USE_HF_MODEL", "false").lower() == "true"
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
PROMPTS_DIR = "src/prompts"


# Initialize OpenAI client (only if not using HF model and API key is available)
client = None
if not USE_HF_MODEL:
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI()
        except ImportError:
            print("OpenAI SDK not installed. OpenAI fallback will not work.")
        except Exception as e:
            print(f"OpenAI client initialization failed: {e}")
    else:
        print("No OpenAI API key found. Will use rule-based approach.")



def load_prompt(version="v2"):
    pfile = f"{PROMPTS_DIR}/reasoner_{version}.txt"
    try:
        with open(pfile, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # Fallback prompt if versioned file is missing
        return "You are a controller that decides the next action. Return compact JSON only."



class Reasoner:
    def __init__(self, version="v2", temperature=0.0):
        self.prompt = load_prompt(version)
        self.version = version
        self.temperature = temperature


    def _call_hf_api(self, prompt, max_tokens=100):
        """Call Hugging Face hosted model via Inference API."""
        if not HUGGINGFACE_API_TOKEN:
            raise RuntimeError("Hugging Face API token not set in .env file. Please add HUGGINGFACE_API_TOKEN=your_token_here to your .env file.")
        
        url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": max_tokens, "temperature": self.temperature},
            "options": {"wait_for_model": True}
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            output = resp.json()
            if isinstance(output, list) and len(output) > 0 and "generated_text" in output[0]:
                return output[0]["generated_text"].strip()
            elif isinstance(output, dict) and "error" in output:
                raise RuntimeError(f"Hugging Face API error: {output['error']}")
            else:
                return str(output)
        except requests.exceptions.RequestException as e:
            print(f"HF API request failed: {e}")
            raise RuntimeError(f"Hugging Face API request failed: {e}")
        except Exception as e:
            print(f"HF API failed: {e}")
            raise


    def _rule_based_decision(self, user_query, kb_hits):
        """Simple rule-based decision making when LLL APIs are not available."""
        query_lower = user_query.lower()

        # 0) Current-events / news detection → prefer Web search (minimal precedence rule)
        news_keywords = ["news", "latest", "today", "headline", "headlines", "breaking", "update", "updates"]
        if any(k in query_lower for k in news_keywords):
            return json.dumps({
                "decision": "TOOL_WEB",
                "summary": "Current-events/news query detected; using Web Search",
                "confidence": 0.9,
                "reason": "Query contains terms indicating current news or updates",
                "tool_args": {"query": user_query}
            })


        # 1) Product/pricing detection → CSV
        csv_keywords = ['price', 'cost', 'buy', 'purchase', 'sku', 'product', 'item', 'available', 'stock']
        needs_csv = any(keyword in query_lower for keyword in csv_keywords)


        # 2) Relevance estimation for KB
        relevant_hits = 0
        if kb_hits:
            query_words = query_lower.split()
            for hit in kb_hits:
                hit_text_lower = hit['text'].lower()
                if any(word in hit_text_lower for word in query_words if len(word) > 3):
                    relevant_hits += 1


        # 3) Confidence heuristic
        if needs_csv:
            confidence = 0.9
        elif relevant_hits > 0:
            confidence = 0.7
        elif kb_hits:
            confidence = 0.3  # Low confidence if hits exist but aren't relevant
        else:
            confidence = 0.1


        if needs_csv:
            return json.dumps({
                "decision": "TOOL_CSV",
                "summary": "Query appears to be about pricing/product information",
                "confidence": confidence,
                "reason": "Query contains keywords suggesting product/pricing data needed",
                "tool_args": {"query": user_query}
            })
        else:
            if relevant_hits > 0:
                summary = f"Query can be answered from knowledge base ({relevant_hits} relevant hits found)"
                reason = "Query appears to be informational and relevant documents found in KB"
            elif kb_hits:
                summary = f"Query may be answered from knowledge base ({len(kb_hits)} hits found, but relevance unclear)"
                reason = "Query appears to be informational but KB content may not be relevant"
            else:
                summary = "No relevant information found in knowledge base"
                reason = "No documents found that match the query"


            return json.dumps({
                "decision": "KB",
                "summary": summary,
                "confidence": confidence,
                "reason": reason,
                "tool_args": {}
            })


    def _call_llm(self, user_query, kb_hits, shared_context=None):
        system = self.prompt
        kb_text = "\n\n".join([f"ID:{h['id']}\nTEXT:{h['text']}" for h in kb_hits]) if kb_hits else "No KB hits."


        # Add context information if available
        context_info = ""
        if shared_context:
            tool_usage = shared_context.get("tool_usage_count", {})
            context_info = f"\nCONTEXT:\nTool usage so far: {tool_usage}\nQuery history length: {len(shared_context.get('query_history', []))}"


        user_msg = f"Query: {user_query}\n\nKB_HITS:\n{kb_text}{context_info}\n\nReturn JSON only."
        prompt = f"{system}\n\n{user_msg}"


        if USE_HF_MODEL:
            try:
                return self._call_hf_api(prompt, max_tokens=100)
            except Exception:
                print("HF model failed, falling back to rule-based approach...")
                return self._rule_based_decision(user_query, kb_hits)


        # OpenAI fallback
        if client is None:
            print("No OpenAI client available, using rule-based approach...")
            return self._rule_based_decision(user_query, kb_hits)


        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user_msg}],
                temperature=self.temperature,
                max_tokens=300
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API failed: {e}")
            print("Falling back to rule-based approach...")
            return self._rule_based_decision(user_query, kb_hits)


    def decide(self, user_query, kb_hits, shared_context=None):
        text = self._call_llm(user_query, kb_hits, shared_context)
        try:
            return json.loads(text.strip())
        except Exception:
            import re
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except:
                    pass
            return {"decision": "KB", "reason": "Could not parse LLM output; default to KB", "tool_args": {}}


    def _rule_based_final_answer(self, user_query, kb_hits, tool_result=None, shared_context=None):
        """Simple rule-based final answer when LLM APIs are not available, with CSV as authoritative for pricing."""
        answer_parts = []
        trace_parts = []
        query_lower = user_query.lower()
        last_decision = (shared_context or {}).get("last_decision", {})
        chose_csv = last_decision.get("decision") == "TOOL_CSV"

        # NEW: If Web was chosen and results exist, render summarized headlines (non-intrusive)
        if last_decision.get("decision") == "TOOL_WEB" and isinstance(tool_result, list) and len(tool_result) > 0:
            lines = []
            for item in tool_result[:5]:
                # Support multiple possible key names used by different web actors
                title = item.get("title") or "Result"
                snippet = item.get("snippet") or item.get("content") or item.get("description") or ""
                title = str(title).strip()
                snippet = str(snippet).strip()
                if snippet:
                    lines.append(f"{title}: {snippet}")
                else:
                    lines.append(f"{title}")
            trace_parts.extend([
                "1. Selected Web Search",
                "2. Retrieved web results",
                "3. Returned summarized headlines"
            ])
            return "\n".join(lines) + "\n\nTrace:\n" + "\n".join(trace_parts)

        # If CSV was chosen for pricing but returned empty, do not surface KB examples as authoritative
        if chose_csv and (not tool_result or (isinstance(tool_result, list) and len(tool_result) == 0)):
            answer_parts.append("No matching product found in catalog (CSV); cannot determine price from knowledge base examples.")
            trace_parts.extend([
                "1. Selected CSV for pricing",
                "2. CSV lookup returned no rows",
                "3. Enforced CSV as authoritative source for prices",
                "4. Responded with not-found"
            ])
            return "\n".join(answer_parts) + "\n\nTrace:\n" + "\n".join(trace_parts)


        # Prefer CSV tool result if available
        if tool_result and isinstance(tool_result, list) and len(tool_result) > 0:
            # Detect list-all products intent (no extra imports)
            list_intent = (
                ("show" in query_lower or "list" in query_lower or "available" in query_lower) and
                ("products" in query_lower or "product" in query_lower or "items" in query_lower or "item" in query_lower)
            )

            if list_intent:
                # Enumerate all matched products
                lines = []
                for row in tool_result:
                    name = row.get("name", "Unknown")
                    sku = row.get("sku") or row.get("SKU") or row.get("Sku")
                    price = row.get("price", "N/A")
                    currency = row.get("currency", "")
                    stock = row.get("stock", "")
                    parts = []
                    if name: parts.append(f"{name}")
                    if sku: parts.append(f"(SKU {sku})")
                    if price != "" and currency:
                        parts.append(f"- {price} {currency}")
                    if stock != "":
                        parts.append(f"- stock {stock}")
                    lines.append(" ".join(parts))
                trace_parts.extend([
                    "1. Selected CSV for catalog listing",
                    "2. Retrieved product rows from CSV",
                    "3. Returned authoritative catalog list from CSV"
                ])
                return "\n".join(lines) + "\n\nTrace:\n" + "\n".join(trace_parts)


            # Non-list intent: keep previous concise single-product output
            first = tool_result[0]
            name = first.get("name", "Unknown")
            sku = first.get("sku") or first.get("SKU") or first.get("Sku")
            price = first.get("price", "N/A")
            currency = first.get("currency", "")
            answer_parts.append(f"{name}: {price} {currency}".strip())
            if sku:
                answer_parts.append(f"SKU: {sku}")
            trace_parts.extend([
                "1. Selected CSV for pricing",
                "2. Retrieved product row from CSV",
                "3. Returned authoritative price from CSV"
            ])
            return "\n".join(answer_parts) + "\n\nTrace:\n" + "\n".join(trace_parts)


        # If not a CSV case, optionally include relevant KB snippets
        relevant_hits = []
        if kb_hits:
            query_words = query_lower.split()
            for hit in kb_hits:
                hit_text_lower = hit['text'].lower()
                if any(word in hit_text_lower for word in query_words if len(word) > 3):
                    relevant_hits.append(hit)


        if relevant_hits:
            answer_parts.append("Based on the knowledge base:")
            for hit in relevant_hits[:2]:
                answer_parts.append(f"- {hit['text'][:200]}...")
            trace_parts.extend([
                f"1. Searched knowledge base and found {len(kb_hits)} documents",
                f"2. Identified {len(relevant_hits)} relevant documents",
                "3. Compiled information to provide this answer"
            ])
            return "\n".join(answer_parts) + "\n\nTrace:\n" + "\n".join(trace_parts)


        if kb_hits:
            answer_parts.append("Found knowledge base documents, but none appear directly relevant to the query.")
            trace_parts.extend([
                f"1. Searched knowledge base and found {len(kb_hits)} documents",
                "2. No relevant documents found for this query",
                "3. Provided a neutral response"
            ])
            return "\n".join(answer_parts) + "\n\nTrace:\n" + "\n".join(trace_parts)


        # Default neutral response
        answer_parts.append("No relevant information found to answer the query.")
        trace_parts.append("1. No KB hits and no tool results; responded neutrally")
        return "\n".join(answer_parts) + "\n\nTrace:\n" + "\n".join(trace_parts)


    def final_answer(self, user_query, kb_hits, tool_result=None, shared_context=None):
        """
        Compose the final answer and a short step-by-step trace.
        If shared_context indicates CSV was chosen for pricing and CSV is empty, avoid surfacing KB prices.
        """
        parts = []
        parts.append("KB_HITS:\n" + ("\n\n".join([f"{h['id']}: {h['text']}" for h in kb_hits]) if kb_hits else "No KB hits"))
        parts.append("TOOL_RESULT:\n" + (json.dumps(tool_result, indent=2) if tool_result else "No tool result"))


        # Optional context like decide()
        context_info = ""
        if shared_context:
            tool_usage = shared_context.get("tool_usage_count", {})
            qhist_len = len(shared_context.get("query_history", []))
            context_info = f"\n\nCONTEXT:\nTool usage so far: {tool_usage}\nQuery history length: {qhist_len}"


        user_msg = f"Query: {user_query}\n\n{chr(10).join(parts)}{context_info}\n\nProvide a final concise answer and a short step-by-step trace of how you arrived at it."
        system_msg = "You are a helpful assistant that produces final answers and traces."


        # If HF model is configured
        if USE_HF_MODEL:
            try:
                return self._call_hf_api(f"{system_msg}\n\n{user_msg}", max_tokens=200)
            except Exception:
                print("HF final_answer failed, falling back to rule-based approach...")
                return self._rule_based_final_answer(user_query, kb_hits, tool_result, shared_context)


        # OpenAI fallback
        if client is None:
            print("No OpenAI client available for final answer, using rule-based approach...")
            return self._rule_based_final_answer(user_query, kb_hits, tool_result, shared_context)


        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system_msg},
                          {"role": "user", "content": user_msg}],
                temperature=0.0,
                max_tokens=400
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"OpenAI final_answer failed: {e}")
            print("Falling back to rule-based approach...")
            return self._rule_based_final_answer(user_query, kb_hits, tool_result, shared_context)
