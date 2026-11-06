"""Indian-news FAKE/REAL: baselines, retrieval of similar known articles, evidence-aware model, evaluation, error analysis."""
import json, re, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from common import *
import matplotlib.pyplot as plt
SEED = 42
d = pd.read_csv("data/dataset_2000.csv", keep_default_na=False)
d["key"] = d.claim.map(lambda t: re.sub(r"[\s.,;:!?\"'()\[\]{}\-|/\\।]+", " ", t.lower()).strip()); assert d.key.is_unique and len(d) == 2000
strat = d.language + "_" + d.y.astype(str)
tr, tmp = train_test_split(d.index, test_size=0.30, stratify=strat, random_state=SEED)
va, te = train_test_split(tmp, test_size=0.50, stratify=strat[tmp], random_state=SEED)
d["split"] = "train"; d.loc[va, "split"] = "dev"; d.loc[te, "split"] = "test"
assert not (set(d.key[tr]) & set(d.key[te])) and not (set(d.key[tr]) & set(d.key[va]))
d[["id", "split"]].to_csv("data/splits.csv", index=False)
TR, VA, TE = d[d.split == "train"], d[d.split == "dev"], d[d.split == "test"]
print("split", len(TR), len(VA), len(TE))
M = lambda: {"NaiveBayes": MultinomialNB(alpha=0.3), "LogReg": LogisticRegression(C=10, max_iter=3000), "LinearSVM": LinearSVC(C=0.5)}
met = lambda y, p: dict(acc=accuracy_score(y, p), macro_f1=f1_score(y, p, average="macro"), precision=precision_score(y, p, average="macro", zero_division=0), recall=recall_score(y, p, average="macro"))
pl = lambda df, p: {l: round(f1_score(df.y[df.language == l], p[(df.language == l).values], average="macro"), 3) for l in sorted(df.language.unique())}
R = {"claim_only": {}, "retrieval": {}, "evidence_aware": {}}

# ---- Phase 3: claim-only ----
cv = Stack(claim_vectorizers()); Xtr, Xva, Xte = cv.fit_transform(TR.claim), cv.transform(VA.claim), cv.transform(TE.claim)
for n, m in M().items():
    m.fit(Xtr, TR.y); p = m.predict(Xte)
    R["claim_only"][n] = {"dev": met(VA.y, m.predict(Xva)), "test": met(TE.y, p), "per_language": pl(TE, p), "confusion": confusion_matrix(TE.y, p).tolist()}
    print("claim-only", n, round(R["claim_only"][n]["test"]["macro_f1"], 3), R["claim_only"][n]["per_language"])
R["claim_only"]["Majority"] = {"test": met(TE.y, np.zeros(len(TE), int))}

# ---- Phase 4/5: evidence = similar already-labelled articles retrieved (TF-IDF cosine) from the TRAIN collection ----
# (these datasets contain no evidence passages, so annotated Recall@k is impossible; we report retrieval label-purity instead)
pool = TR.assign(owner=TR.id, text=TR.claim)[["owner", "text"]]; ret = Retriever(pool); ylab = TR.set_index("id").y
def neigh(df, self_exclude):
    ex = df.id.tolist() if self_exclude else None
    out = []
    for (idx, s) in ret.search(df.claim.tolist(), k=5, exclude_owner=ex):
        out.append((ret.owner[idx], s))
    return out
def feats(nb):
    F = []
    for ids, s in nb:
        y = ylab.loc[ids].values; w = np.maximum(s, 1e-6)
        F.append([(w * y).sum() / w.sum(), y.mean(), s.max(), s.mean(), (y[0] if len(y) else .5)])
    return np.array(F)
Ntr, Nva, Nte = neigh(TR, True), neigh(VA, False), neigh(TE, False)
Ftr, Fva, Fte = feats(Ntr), feats(Nva), feats(Nte)
# retrieval quality: does top-5 neighbour majority vote match the true label?  (+ per language)
vote = (Fte[:, 1] >= 0.5).astype(int)
R["retrieval"] = {"pool_size": len(pool), "kNN_vote_acc_test": round(float(accuracy_score(TE.y, vote)), 3), "kNN_vote_macroF1_test": round(float(f1_score(TE.y, vote, average="macro")), 3),
                  "top1_same_language_rate": round(float(np.mean([TR.set_index("id").language.loc[i[0][0]] == l for i, l in zip(Nte, TE.language)])), 3), "per_language": pl(TE, vote)}
print("retrieval", R["retrieval"])
rng = np.random.RandomState(0)
for n, m in M().items():
    g = lambda X, F: sp.hstack([X, sp.csr_matrix(F * 3)]).tocsr()
    m.fit(g(Xtr, Ftr), TR.y); p = m.predict(g(Xte, Fte))
    R["evidence_aware"][n] = {"dev": met(VA.y, m.predict(g(Xva, Fva))), "test": met(TE.y, p), "per_language": pl(TE, p), "confusion": confusion_matrix(TE.y, p).tolist()}
    print("claim+retrieved", n, round(R["evidence_aware"][n]["test"]["macro_f1"], 3))
m = LogisticRegression(C=10, max_iter=3000).fit(sp.hstack([Xtr, sp.csr_matrix(Ftr[rng.permutation(len(Ftr))] * 3)]).tocsr(), TR.y)
R["evidence_aware"]["random_evidence_control_LogReg"] = {"test": met(TE.y, m.predict(sp.hstack([Xte, sp.csr_matrix(Fte[rng.permutation(len(Fte))] * 3)]).tocsr()))}

# ---- Phase 6: robustness / bias ----
loo = {}
for l in sorted(d.language.unique()):
    a, b = d[d.language != l], d[d.language == l]; v = Stack(claim_vectorizers())
    loo[l] = round(f1_score(b.y, LogisticRegression(C=10, max_iter=3000).fit(v.fit_transform(a.claim), a.y).predict(v.transform(b.claim)), average="macro"), 3)
R["leave_one_language_out_LogReg"] = loo; print("LOLO", loo)
S = {k: [] for k in M()}
for a, b in RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1).split(d, strat):
    v = Stack(claim_vectorizers()); Xa, Xb = v.fit_transform(d.claim.iloc[a]), v.transform(d.claim.iloc[b])
    for k, m in M().items(): S[k].append(f1_score(d.y.iloc[b], m.fit(Xa, d.y.iloc[a]).predict(Xb), average="macro"))
R["cv_5x2_claim_only"] = {k: {"mean": round(float(np.mean(v)), 3), "std": round(float(np.std(v)), 3)} for k, v in S.items()}; print(R["cv_5x2_claim_only"])
# source-bias check on English: how well does the news SITE alone predict the label?
en = d[d.language == "en"]; ct = pd.crosstab(en.site, en.y)
site_acc = (ct.max(axis=1).sum() / ct.values.sum())
R["english_site_only_accuracy_upper_bound"] = round(float(site_acc), 3)
R["english_sites_by_label"] = ct.assign(n=ct.sum(axis=1)).sort_values("n", ascending=False).head(8).to_dict("index")
best = max(M(), key=lambda n: R["claim_only"][n]["dev"]["macro_f1"]); m = M()[best].fit(Xtr, TR.y); E = TE.assign(pred=m.predict(Xte)); E = E[E.pred != E.y]
R["error_analysis"] = {"model": best, "n_errors": len(E), "error_rate_by_language": (E.groupby("language").size() / TE.groupby("language").size()).round(3).fillna(0).to_dict(),
                       "fake_predicted_real": int((E.y == 0).sum()), "real_predicted_fake": int((E.y == 1).sum())}
E[["language", "site", "claim", "label_name", "pred"]].head(40).to_csv("results/error_examples.csv", index=False)
wv = cv.vecs[0]; lr = LogisticRegression(C=10, max_iter=3000).fit(wv.transform(TR.claim), TR.y); nm = np.array(wv.get_feature_names_out()); o = np.argsort(lr.coef_[0])
R["top_fake_tokens"], R["top_real_tokens"] = nm[o[:25]].tolist(), nm[o[-25:]].tolist()
with open("results/results.json", "w", encoding="utf-8") as f:
    json.dump(R, f, indent=1, ensure_ascii=False, default=float)

# Display confusion matrices for all three models
from sklearn.metrics import ConfusionMatrixDisplay

for name in ["NaiveBayes", "LogReg", "LinearSVM"]:
    cm = np.array(R["claim_only"][name]["confusion"])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["FAKE", "REAL"]
    )

    display.plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()

    filename = f"results/confusion_matrix_{name}.png"
    plt.savefig(filename, dpi=300)
    plt.show()

    print(f"{name} confusion matrix:")
    print(cm)

models = ["Naive Bayes", "Logistic Regression", "Linear SVM"]
scores = [0.900, 0.903, 0.897]

plt.bar(models, scores)
plt.ylabel("Macro-F1 Score")
plt.title("Fake News Detection: Model Comparison")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("results/model_comparison.png", dpi=300)
plt.show()