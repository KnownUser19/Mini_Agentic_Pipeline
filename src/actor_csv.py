# src/actor_csv.py
import os
import time
import re
import pandas as pd
from typing import List, Tuple, Dict, Optional


class CSVActor:
    def __init__(self, csv_path: str = "prices.csv"):
        self.csv_path = csv_path
        self.df = None
        self.load_error = None
        self._load_df()

    def _load_df(self):
        """Robustly load a 'CSV' that may be comma, tab, or whitespace delimited."""
        if not os.path.exists(self.csv_path):
            self.load_error = f"CSV not found: {self.csv_path}"
            self.df = None
            return

        # Try multiple strategies
        candidates = [
            dict(sep=None, engine="python"),          # sniff delimiter
            dict(delim_whitespace=True, engine="python"),  # whitespace-delimited
            dict(sep=",", engine="python"),           # classic CSV
            dict(sep=";", engine="python"),           # semicolon CSV
        ]
        last_err = None
        for kwargs in candidates:
            try:
                df = pd.read_csv(self.csv_path, **kwargs, on_bad_lines="skip")
                if df is not None and df.shape[1] >= 2:
                    self.df = self._normalize_df(df)
                    self.load_error = None
                    return
            except Exception as e:
                last_err = e

        # Fallback: try manual whitespace split
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if lines:
                # Split on runs of whitespace
                rows = [re.split(r"\s{2,}|\t+|\s+", ln) for ln in lines]
                # If first row looks like a header, use it
                header = [h.strip().lower() for h in rows[0]]
                if "sku" in header:
                    body = rows[1:]
                    df = pd.DataFrame(body, columns=header[:len(body[0])])
                else:
                    # Synthesize minimal header
                    df = pd.DataFrame(rows, columns=[f"col{i}" for i in range(len(rows[0]))])
                self.df = self._normalize_df(df)
                self.load_error = None
                return
        except Exception as e:
            last_err = e

        self.df = None
        self.load_error = f"Failed to load CSV with autodetection. Last error: {last_err}"

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Normalize column names
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]

        # If likely a single wide column, try to parse into fields
        if df.shape[1] == 1 and "sku" not in df.columns:
            # Attempt to expand using common delimiters inside the single column
            col = df.columns[0]
            expanded = df[col].str.split(r"[,\t;]\s*|\s{2,}", expand=True)
            if isinstance(expanded, pd.DataFrame) and expanded.shape[1] > 1:
                df = expanded
                df.columns = [f"col{i}" for i in range(df.shape[1])]
                df.columns = [c.strip().lower() for c in df.columns]

        # Try to map probable columns to canonical names
        rename_map = {}
        for c in list(df.columns):
            lc = c.lower().strip()
            if lc in {"id", "sku_id"}:
                rename_map[c] = "sku"
            elif lc in {"product", "item", "title"}:
                rename_map[c] = "name"
        if rename_map:
            df = df.rename(columns=rename_map)

        # If canonical columns exist but with case/space differences, standardize
        # Try to identify columns by content if names are unknown
        cols = set(df.columns)
        # Ensure sku column exists if possible
        if "sku" not in cols:
            # Heuristic: a column that matches patterns like ABC123 in most rows
            for c in df.columns:
                sample = df[c].astype(str).str.match(r"^[A-Za-z]{2,}\d{2,}$", na=False).mean()
                if sample > 0.3:  # heuristic threshold
                    df = df.rename(columns={c: "sku"})
                    break

        # Ensure name column
        if "name" not in df.columns and "col1" in df.columns:
            df = df.rename(columns={"col1": "name"})

        # Trim whitespace from string cells
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.strip()

        # Lowercase-only search columns keep original display columns intact via copies
        # Create helper lowercase series for search
        if "sku" in df.columns:
            df["_sku_lc"] = df["sku"].astype(str).str.strip().str.lower()
        if "name" in df.columns:
            df["_name_lc"] = df["name"].astype(str).str.strip().str.lower()
        if "description" in df.columns:
            df["_desc_lc"] = df["description"].astype(str).str.strip().str.lower()

        return df

    def _is_sku_token(self, q: str) -> Optional[str]:
        q = q.strip()
        # Extract explicit 'sku XYZ' or bare token like 'PEN456'
        m = re.search(r"\bsku[-\s:]*([A-Za-z0-9\-]+)\b", q, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # Bare token heuristic
        if re.fullmatch(r"[A-Za-z]{2,}\d{2,}(-[A-Za-z0-9]+)?", q):
            return q.upper()
        return None

    def lookup(self, query: str) -> Tuple[List[Dict], float]:
        """
        Lookup by SKU or query text.
        Returns (rows_as_dicts, latency_seconds)
        """
        t0 = time.perf_counter()
        if self.df is None:
            # Failed to load; return empty but keep latency
            return [], time.perf_counter() - t0

        q = query.strip()
        q_clean = re.sub(r"[^A-Za-z0-9\s\-:]", "", q).strip()
        q_lc = q_clean.lower()

        # Show all / list products intent
        # CHANGED: accept both 'product' and 'products'
        if re.search(r"\b(show|list|available)\b.*\b(products?|items?)\b", q_lc):
            rows = self.df.copy()
            return self._rows_to_dicts(rows), time.perf_counter() - t0

        # If SKU token is present, do exact equality (case-insensitive)
        sku_tok = self._is_sku_token(q_clean)
        if sku_tok and "sku" in self.df.columns:
            rows = self.df[self.df["_sku_lc"] == sku_tok.lower()]
            return self._rows_to_dicts(rows), time.perf_counter() - t0

        # Otherwise, perform substring search across relevant columns
        mask = None
        for col in ["_sku_lc", "_name_lc", "_desc_lc"]:
            if col in self.df.columns:
                part = self.df[col].str.contains(q_lc, na=False)
                mask = part if mask is None else (mask | part)

        rows = self.df[mask] if mask is not None else self.df.iloc[0:0]
        return self._rows_to_dicts(rows), time.perf_counter() - t0

    def _rows_to_dicts(self, rows: pd.DataFrame) -> List[Dict]:
        if rows is None or rows.empty:
            return []
        # Prefer canonical display columns
        display_cols = [c for c in ["sku", "name", "category", "price", "currency", "stock", "description"] if c in rows.columns]
        if not display_cols:
            display_cols = [c for c in rows.columns if not c.startswith("_")]
        return rows[display_cols].to_dict(orient="records")
