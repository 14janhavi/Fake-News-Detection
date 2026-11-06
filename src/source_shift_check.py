"""Honest check for English: train on some news SITES, test on SITES never seen (group split). Sites are almost label-pure,
so a random split lets the model cheat by recognising the outlet's writing style."""
import numpy as np, pandas as pd, json
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from common import *
d = pd.read_csv("data/dataset_2000.csv", keep_default_na=False); en = d[d.language == "en"].reset_index(drop=True)
out = {}
for name, mk in {"NaiveBayes": lambda: MultinomialNB(alpha=0.3), "LogReg": lambda: LogisticRegression(C=10, max_iter=3000)}.items():
    s = []
    for a, b in GroupKFold(n_splits=5).split(en, en.y, groups=en.site):
        v = Stack(claim_vectorizers()); m = mk().fit(v.fit_transform(en.claim.iloc[a]), en.y.iloc[a])
        if en.y.iloc[b].nunique() > 1: s.append(f1_score(en.y.iloc[b], m.predict(v.transform(en.claim.iloc[b])), average="macro"))
    out[name] = {"mean": round(float(np.mean(s)), 3), "folds": [round(x, 3) for x in s]}
print(out); json.dump(out, open("results/english_unseen_source.json", "w"), indent=1)
