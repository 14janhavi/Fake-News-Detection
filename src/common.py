"""Shared code: text cleaning, features (trained from scratch - NO pretrained models), retrieval, evidence features."""
import re, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

TOK = r"[^\s.,;:!?\"'()\[\]{}\-\u2013\u2014|/\\\u0964\u2018\u2019\u201c\u201d]+"
EV = [f"evidence_{i}" for i in range(1, 6)]

def clean(t):
    t = re.sub(r"https?://\S+|www\.\S+", " ", str(t).lower())
    t = re.sub(r"\d{4}[/-]\d{2}[/-]\d{2}\s*[—-]", " ", t)       # leading snippet dates
    return re.sub(r"\s+", " ", t).strip()

def claim_vectorizers():
    # word 1-2grams + character 2-5grams (char n-grams make it work across scripts/languages without a tokenizer)
    return [TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, preprocessor=clean, token_pattern=TOK),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, preprocessor=clean, max_features=200000)]

class Stack:
    """fit several vectorizers on the same text and hstack"""
    def __init__(self, vecs): self.vecs = vecs
    def fit(self, X): [v.fit(X) for v in self.vecs]; return self
    def transform(self, X): return sp.hstack([v.transform(X) for v in self.vecs]).tocsr()
    def fit_transform(self, X): return self.fit(X).transform(X)

def explode_evidence(df):
    rows = []
    for r in df.itertuples():
        for j, c in enumerate(EV):
            t = getattr(r, c)
            if isinstance(t, str) and len(t.strip()) > 10: rows.append((r.id, j, t))
    return pd.DataFrame(rows, columns=["owner", "rank", "text"])

class Retriever:
    """TF-IDF + cosine retrieval over a FIXED evidence collection (the pool)."""
    def __init__(self, pool_df):
        self.pool = pool_df.reset_index(drop=True)
        self.vec = Stack([TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, preprocessor=clean, token_pattern=TOK, min_df=1),
                          TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, preprocessor=clean, min_df=2, max_features=300000)])
        self.M = normalize(self.vec.fit_transform(self.pool.text))
        self.owner = self.pool.owner.values
    def search(self, queries, k=5, exclude_owner=None):
        Q = normalize(self.vec.transform(queries)); S = (Q @ self.M.T).toarray()
        res = []
        for i in range(S.shape[0]):
            s = S[i].copy()
            if exclude_owner is not None: s[self.owner == exclude_owner[i]] = -1
            idx = np.argsort(-s)[:k]; res.append((idx, s[idx]))
        return res

def sim_feats(claim_vec_rows, ev_vec_rows):
    """cosine(claim, each evidence) -> [max, mean, min, top2 mean] per claim"""
    out = []
    for c, E in zip(claim_vec_rows, ev_vec_rows):
        if E is None or E.shape[0] == 0: out.append([0, 0, 0, 0]); continue
        s = np.asarray((E @ c.T).todense()).ravel(); s.sort()
        out.append([s[-1], s.mean(), s[0], s[-2:].mean()])
    return np.array(out)
