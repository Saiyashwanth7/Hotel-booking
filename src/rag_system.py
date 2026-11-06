# src/rag_system.py
import os
import re
import time
from datetime import datetime

import numpy as np
import pycountry

# "tfidf" (default): tiny and fast, no torch needed, fits free hosting (512 MB RAM).
# "transformer": sentence-transformers + FAISS, better semantic search but needs ~1 GB+ RAM.
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "tfidf")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
]
COUNTRY_ALIASES = {
    "uk": "GBR", "britain": "GBR", "england": "GBR", "usa": "USA", "america": "USA",
    "russia": "RUS", "holland": "NLD",
}


def _load_embedder(name):
    """Only used when EMBEDDING_BACKEND=transformer."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def country_name(code):
    c = pycountry.countries.get(alpha_3=str(code))
    return c.name if c else str(code)


def _build_country_lookup():
    lookup = {}
    for c in pycountry.countries:
        lookup[c.name.lower()] = c.alpha_3
        if hasattr(c, "common_name"):
            lookup[c.common_name.lower()] = c.alpha_3
    lookup.update(COUNTRY_ALIASES)
    return lookup


class RAGSystem:
    def __init__(self, data, llm_model=None, api_key=None, base_url=None):
        self.data = data
        # Any OpenAI-compatible provider works. Defaults to OpenRouter (free models).
        self.llm_model = llm_model or os.environ.get("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        # Free models come and go, so try these in order if the main model fails
        fallbacks = os.environ.get("LLM_FALLBACK_MODELS", "openrouter/free")
        self.fallback_models = [m.strip() for m in fallbacks.split(",") if m.strip()]
        self.embedder = None
        self.index = None
        self.documents = []
        self.query_history = []
        self.country_lookup = _build_country_lookup()

    # ------------------------------------------------------------------ index
    def _stats_text(self, df):
        n = len(df)
        canc = int(df["is_canceled"].sum())
        return (
            f"{n} bookings, {canc} cancellations (cancellation rate {canc / n * 100:.2f}%), "
            f"total revenue {df['revenue'].sum():,.2f}, average daily rate (ADR) {df['adr'].mean():.2f}, "
            f"average lead time {df['lead_time'].mean():.2f} days, "
            f"average stay {df['total_nights'].mean():.2f} nights."
        )

    def build_documents(self):
        """Aggregated summary documents. Much smaller and more useful than one doc per booking."""
        df = self.data
        hotels = [("all hotels", df)] + [(h, g) for h, g in df.groupby("hotel")]
        docs = []

        # Overall + yearly + monthly summaries per hotel
        for label, g in hotels:
            docs.append(f"Overall summary for {label}: {self._stats_text(g)}")
            for year, gy in g.groupby(g["arrival_date"].dt.year):
                docs.append(f"Summary for {label} in {year}: {self._stats_text(gy)}")
            for period, gm in g.groupby(g["arrival_date"].dt.to_period("M")):
                month = MONTH_NAMES[period.month - 1].capitalize()
                docs.append(f"Summary for {label} in {month} {period.year}: {self._stats_text(gm)}")

        # Country summaries (all hotels, plus per hotel for larger markets)
        for code, g in df.groupby("country"):
            docs.append(f"Guests from {country_name(code)} ({code}) at all hotels: {self._stats_text(g)}")
            if len(g) >= 200:
                for h, gh in g.groupby("hotel"):
                    docs.append(f"Guests from {country_name(code)} ({code}) at {h}: {self._stats_text(gh)}")

        # Segment summaries
        for col, name in [("market_segment", "market segment"), ("customer_type", "customer type"),
                          ("deposit_type", "deposit type")]:
            for value, g in df.groupby(col):
                docs.append(f"Bookings with {name} {value}: {self._stats_text(g)}")

        # Top cancellation countries
        top = df[df["is_canceled"] == 1]["country"].value_counts().head(10)
        listing = ", ".join(f"{country_name(c)} ({c}): {n}" for c, n in top.items())
        docs.append(f"Countries with the highest number of cancellations: {listing}.")

        self.documents = docs
        return docs

    def build_index(self):
        docs = self.build_documents()
        if EMBEDDING_BACKEND == "transformer":
            import faiss
            if self.embedder is None:
                self.embedder = _load_embedder(EMBEDDING_MODEL)
            emb = np.asarray(
                self.embedder.encode(docs, normalize_embeddings=True, show_progress_bar=False),
                dtype="float32",
            )
            self.index = faiss.IndexFlatIP(emb.shape[1])
            self.index.add(emb)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
            self.index = self.vectorizer.fit_transform(docs)  # L2-normalised sparse matrix
        return len(docs)

    def retrieve(self, query, top_k=6):
        if self.index is None:
            return []
        if EMBEDDING_BACKEND == "transformer":
            q = np.asarray(self.embedder.encode([query], normalize_embeddings=True), dtype="float32")
            _, idx = self.index.search(q, top_k)
            ids = [i for i in idx[0] if i >= 0]
        else:
            scores = (self.index @ self.vectorizer.transform([query]).T).toarray().ravel()
            ids = [int(i) for i in np.argsort(-scores)[:top_k] if scores[i] > 0]
        return [self.documents[i] for i in ids]

    # ---------------------------------------------------- computed (exact) answers
    def _parse(self, q):
        ql = q.lower()
        hotel = None
        if "resort" in ql:
            hotel = "Resort Hotel"
        elif "city hotel" in ql or "city hotels" in ql:
            hotel = "City Hotel"

        country = None
        best = 0
        for name, code in self.country_lookup.items():
            if re.search(rf"\b{re.escape(name)}\b", ql) and len(name) > best:
                country, best = code, len(name)

        month = next((i + 1 for i, m in enumerate(MONTH_NAMES) if re.search(rf"\b{m}\b", ql)), None)
        year_match = re.search(r"\b(2015|2016|2017)\b", ql)
        year = int(year_match.group(1)) if year_match else None
        return ql, hotel, country, month, year

    def direct_answer(self, question):
        """Exact answers computed with pandas for common metrics. Returns None if not applicable."""
        ql, hotel, country, month, year = self._parse(question)
        both = "resort" in ql and bool(re.search(r"\bcity\b", ql))
        compare_words = bool(re.search(r"\b(compare|comparison|versus|vs|higher|lower|better|worse|difference)\b", ql))
        asks_hotel_type = bool(re.search(r"which (hotel|type)|hotel type", ql))

        if both or (asks_hotel_type and compare_words):
            # Compare the two hotels: answer the metric for each one.
            lines = []
            for h in ["Resort Hotel", "City Hotel"]:
                ans = self._answer_for(ql, h, country, month, year)
                if ans is None:
                    return None
                lines.append(ans)
            return " ".join(lines)
        return self._answer_for(ql, hotel, country, month, year)

    def _answer_for(self, ql, hotel, country, month, year):
        df = self.data
        scope = []
        if hotel:
            df = df[df["hotel"] == hotel]
            scope.append(hotel)
        if country:
            df = df[df["country"] == country]
            scope.append(f"guests from {country_name(country)}")
        scope_txt = f" ({', '.join(scope)})" if scope else ""
        if df.empty:
            return "There are no bookings matching that filter in the dataset."

        if "revenue" in ql and re.search(r"\b(highest|most|best|peak)\b", ql) and "month" in ql:
            d = df[df["arrival_date"].dt.year == year] if year else df
            if d.empty:
                return "There are no bookings for that period in the dataset."
            by_month = d.groupby(d["arrival_date"].dt.to_period("M"))["revenue"].sum()
            best = by_month.idxmax()
            when = f" in {year}" if year else ""
            return (f"The month with the highest revenue{when}{scope_txt} was "
                    f"{MONTH_NAMES[best.month - 1].capitalize()} {best.year} with a revenue of {by_month.max():,.2f}.")

        if "revenue" in ql:
            d = df
            when = ""
            if year:
                d = d[d["arrival_date"].dt.year == year]
                when = str(year)
            if month:
                d = d[d["arrival_date"].dt.month == month]
                when = f"{MONTH_NAMES[month - 1].capitalize()} {when}".strip()
            if d.empty:
                return "There are no bookings for that period in the dataset."
            when_txt = f" for {when}" if when else ""
            return (f"The total revenue{when_txt}{scope_txt} is {d['revenue'].sum():,.2f} "
                    f"(ADR x nights, including canceled bookings).")

        if "cancellation" in ql and re.search(r"\b(highest|most|top|maximum)\b", ql) and \
                re.search(r"\b(countr\w*|locations?|nationalit\w*)\b", ql):
            top = df[df["is_canceled"] == 1]["country"].value_counts().head(5)
            listing = ", ".join(f"{country_name(c)} ({n:,})" for c, n in top.items())
            return f"The countries with the highest number of cancellations{scope_txt} are: {listing}."

        if "cancellation rate" in ql or "cancelation rate" in ql or "cancel rate" in ql:
            rate = df["is_canceled"].mean() * 100
            return f"The cancellation rate{scope_txt} is {rate:.2f}% ({int(df['is_canceled'].sum()):,} of {len(df):,} bookings)."

        if "lead time" in ql:
            return f"The average booking lead time{scope_txt} is {df['lead_time'].mean():.2f} days."

        if re.search(r"average (price|daily rate|rate|adr)|\badr\b", ql):
            return f"The average price (ADR) per booking{scope_txt} is {df['adr'].mean():.2f}."

        if re.search(r"how many bookings|total bookings|number of bookings", ql):
            return f"There are {len(df):,} bookings{scope_txt} in the dataset."

        return None

    # --------------------------------------------------------------------- LLM
    def _call_llm(self, prompt):
        import requests
        messages = [
            {"role": "system", "content": "You answer questions about a hotel booking dataset using ONLY "
             "the provided context. If the context does not contain the answer, say you do not have "
             "enough information. Be concise and quote numbers exactly as given."},
            {"role": "user", "content": prompt},
        ]
        errors = []
        for model in [self.llm_model] + self.fallback_models:
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "X-Title": "Hotel Booking RAG",  # optional, shown on OpenRouter dashboard
                    },
                    json={"model": model, "messages": messages, "max_tokens": 300, "temperature": 0.1},
                    timeout=60,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"{resp.status_code}: {resp.text[:200]}")
                content = resp.json()["choices"][0]["message"]["content"]
                if not content or not content.strip():
                    raise RuntimeError("empty response")
                return content.strip()
            except Exception as e:
                errors.append(f"{model} -> {e}")
        raise RuntimeError(" | ".join(errors))

    # ------------------------------------------------------------------ public
    def answer_question(self, question):
        start = time.time()
        docs = self.retrieve(question)
        source = "computed"
        answer = self.direct_answer(question)

        if answer is None:
            context = "\n".join(f"- {d}" for d in docs)
            prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            try:
                if not self.api_key:
                    raise RuntimeError("OPENROUTER_API_KEY is not set")
                answer = self._call_llm(prompt)
                source = "rag_llm"
            except Exception as e:  # LLM unavailable: still return something useful
                answer = ("The LLM is not available right now, so here is the most relevant data I found:\n"
                          + context)
                source = "retrieval_only"
                print(f"LLM call failed: {e}")

        elapsed = (time.time() - start) * 1000
        self.query_history.append({
            "question": question,
            "answer": answer,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        })
        return {
            "question": question,
            "answer": answer,
            "retrieved_documents": docs,
            "metadata": {"source": source, "processing_time_ms": round(elapsed, 2)},
        }

    def get_query_history(self):
        return self.query_history
