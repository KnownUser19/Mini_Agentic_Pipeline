# utils.py - logging and simple helpers
import time, json
from typing import Dict

def now_ts():
    return time.time()

def make_trace_entry(step:str, details:Dict):
    return {"ts": now_ts(), "step": step, "details": details}

def pretty_print_trace(trace):
    out = []
    for e in trace:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e["ts"]))
        out.append(f"[{ts}] {e['step']}: {e['details']}")
    return "\n".join(out)
