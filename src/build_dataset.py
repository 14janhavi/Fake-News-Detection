"""Build a 2,000-item INDIAN-news FAKE/REAL dataset: English (IFND) + Bengali, Assamese, Odia, Bhojpuri.
Usage: python src/build_dataset.py <dir containing IFND.csv and the four *_dataset.csv files>"""
import sys, re, pandas as pd
src = sys.argv[1]
PER_CLASS, MAXCH = 200, 300       # 5 languages x 2 classes x 200 = 2000; long articles cut to first 300 chars (headline + lede)
def norm(t): return re.sub(r"[\s.,;:!?\"'()\[\]{}\-|/\\\u0964]+", " ", str(t).lower()).strip()   # Indic-safe (keeps vowel signs)
frames = []
e = pd.read_csv(f"{src}/IFND.csv", encoding="latin1")
e = e.rename(columns={"Statement": "claim", "Web": "site"}); e["y"] = (e.Label.str.upper() == "TRUE").astype(int)
frames.append(e.assign(language="en")[["language", "site", "claim", "y"]])
for code, f in [("bn", "bengali"), ("as", "assamese"), ("or", "odia"), ("bho", "bhojpuri")]:
    x = pd.read_csv(f"{src}/{f}_dataset.csv").rename(columns={"text": "claim"})
    x["y"] = (x.label == "Real").astype(int); x["site"] = "n/a"
    frames.append(x.assign(language=code)[["language", "site", "claim", "y"]])
a = pd.concat(frames, ignore_index=True).dropna(subset=["claim"])
a["claim"] = a.claim.astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str[:MAXCH]
a = a[a.claim.str.len() >= 25]
a["key"] = a.claim.map(norm)
c = a.groupby("key").y.nunique(); a = a[~a.key.isin(c[c > 1].index)]      # conflicting labels
a["k60"] = a.key.str[:60]; a = a.drop_duplicates("key").drop_duplicates("k60")         # exact + near-duplicate (same first 60 chars)
out = [a[(a.language == l) & (a.y == y)].sample(PER_CLASS, random_state=42) for l in a.language.unique() for y in (0, 1)]
d = pd.concat(out).reset_index(drop=True); d.insert(0, "id", range(len(d)))
d["label_name"] = d.y.map({0: "FAKE", 1: "REAL"})
d[["id", "language", "site", "claim", "label_name", "y"]].to_csv("data/dataset_2000.csv", index=False)
print(len(d)); print(pd.crosstab(d.language, d.label_name)); print(d.groupby("language").claim.apply(lambda s: s.str.len().mean()).round(0))
print(d[d.language == "en"].site.value_counts().to_string())
