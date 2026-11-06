"""Train the deployment model from scratch (no pretrained weights) on train+dev; test split untouched."""
import os, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from common import *
os.makedirs("models", exist_ok=True)
d = pd.read_csv("data/dataset_2000.csv", keep_default_na=False).merge(pd.read_csv("data/splits.csv"), on="id")
fit = d[d.split != "test"]
v = Stack(claim_vectorizers()); clf = LogisticRegression(C=10, max_iter=3000).fit(v.fit_transform(fit.claim), fit.y)
ret = Retriever(pd.DataFrame({"owner": d.id, "text": d.claim}))
joblib.dump({"vec": v, "clf": clf, "retriever": ret, "meta": d.set_index("id")[["label_name", "language"]]}, "models/fakenews_model.joblib", compress=3)
print("saved; trained on", len(fit), "Indian-news items")
