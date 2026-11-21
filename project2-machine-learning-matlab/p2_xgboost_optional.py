"""Optional: real XGBoost on the same level-6 features as MATLAB.

MATLAB has no XGBoost, so the project's "GradientBoosting" model is
LogitBoost (gradient boosting of trees with the logistic loss). This
script trains real XGBoost on exactly the same features and the same
train/test split, so the two can be compared in the report.

Run after RUN_ME (it needs datasets/level6_features.csv):
    pip install xgboost scikit-learn pandas
    python p2_xgboost_optional.py
Then run RUN_FINISH with START_OVERRIDE = 7 (or just send the file
results/p2_xgboost_results.txt together with the results zip).
"""
import os
import time
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

here = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(here, "datasets", "level6_features.csv"))
is_test = df.pop("isTest").astype(str).str.lower().isin(["1", "true"])
labels = df.pop("label").astype(str)
classes = sorted(labels.unique())
y = labels.map({c: i for i, c in enumerate(classes)}).values
X = df.values
Xtr, Xte, ytr, yte = X[~is_test], X[is_test], y[~is_test], y[is_test]

model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                      subsample=0.9, colsample_bytree=0.8,
                      tree_method="hist", n_jobs=-1)
t0 = time.perf_counter()
model.fit(Xtr, ytr)
train_s = time.perf_counter() - t0
t0 = time.perf_counter()
yp = model.predict(Xte)
infer_us = (time.perf_counter() - t0) / len(yte) * 1e6
size_kb = len(pickle.dumps(model)) / 1024

lines = [
    "XGBoost (Python) on level-6 features",
    f"  classes: {', '.join(classes)}",
    f"  train frames {len(ytr)}, test frames {len(yte)}",
    f"  accuracy {accuracy_score(yte, yp):.4f}",
    f"  macro F1 {f1_score(yte, yp, average='macro'):.4f}",
    f"  training time {train_s:.2f} s, inference {infer_us:.1f} us/frame, size {size_kb:.0f} kB",
    "  confusion matrix (rows = true):",
]
for c, row in zip(classes, confusion_matrix(yte, yp)):
    lines.append(f"    {c:>10s} " + " ".join(f"{v:4d}" for v in row))
text = "\n".join(lines)
print(text)
os.makedirs(os.path.join(here, "results"), exist_ok=True)
with open(os.path.join(here, "results", "p2_xgboost_results.txt"), "w") as f:
    f.write(text + "\n")
