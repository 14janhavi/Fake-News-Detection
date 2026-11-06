"""Predict whether an UNSEEN headline / claim / news text / URL looks FAKE or REAL.
  python predict.py --text "Some headline or claim in any language"
  python predict.py --url https://example.com/news/article
  python predict.py            (interactive)
The model is a TF-IDF + Naive Bayes classifier trained from scratch (no pretrained weights) on Indian news: English, Bengali, Assamese, Odia, Bhojpuri."""
import sys, os, re, argparse, joblib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import common  # needed for joblib unpickling

def fetch_url(url):
    import requests; from bs4 import BeautifulSoup
    r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"}); r.raise_for_status()
    s = BeautifulSoup(r.text, "html.parser")
    for t in s(["script", "style", "nav", "footer", "header", "aside"]): t.decompose()
    title = (s.find("meta", property="og:title") or {}).get("content") or (s.title.string if s.title else "")
    paras = [p.get_text(" ", strip=True) for p in s.find_all("p") if len(p.get_text(strip=True)) > 60]
    return (title or "").strip(), " ".join(paras[:4])

def predict(text, M, topk=3):
    X = M["vec"].transform([text]); p = M["clf"].predict_proba(X)[0]; fake, real = p[0], p[1]
    ev = M["retriever"].search([text], k=topk)[0]
    related = [(round(float(s), 3), M["meta"].label_name.loc[M["retriever"].owner[i]], M["retriever"].pool.text.values[i][:140]) for i, s in zip(*ev)]
    return fake, real, related

def report(text, M):
    fake, real, related = predict(text, M)
    label = "REAL (credible)" if real >= fake else "FAKE (likely false)"; conf = max(fake, real)
    print(f"\nInput : {text[:200]}\nResult: {label}   P(fake)={fake:.2f}  P(real)={real:.2f}")
    if conf < 0.60: print("Note  : LOW CONFIDENCE - treat as 'unsure'; verify with a fact-checking site.")
    if len(text.split()) < 5: print("Note  : very short input; accuracy drops.")
    print("Most similar known Indian news items (TF-IDF cosine) and their labels:")
    for s, l, t in related: print(f"   [{s}] {l}: {t}")
    print("Caution: the model judges wording/style learned from Indian news + fact-check articles; it does NOT verify facts.")
    print("         Expect ~90% on the dataset's own sources but only ~75% on news from outlets it never saw.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--text"); ap.add_argument("--url"); a = ap.parse_args()
    M = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "fakenews_model.joblib"))
    if a.url:
        title, body = fetch_url(a.url); print("Fetched title:", title); report((title + ". " + body).strip(), M)
    elif a.text: report(a.text, M)
    else:
        while True:
            t = input("\nHeadline / claim / URL (blank to quit): ").strip()
            if not t: break
            if t.startswith("http"):
                title, body = fetch_url(t); t = (title + ". " + body).strip()
            report(t, M)
