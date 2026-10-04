"""Final-stage score fusion utility.
Input CSV: text_sim,image_sim,label.
Tune alpha and threshold on validation data.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

def tune(df):
    y=df["label"].to_numpy()
    ts=df["text_sim"].to_numpy(float)
    ims=df["image_sim"].to_numpy(float)
    best=None
    for alpha in np.linspace(0,1,21):
        score=alpha*ts+(1-alpha)*ims
        for threshold in np.linspace(0.05,0.95,181):
            pred=score>=threshold
            f=f1_score(y,pred)
            if best is None or f>best["f1"]:
                best={"alpha":float(alpha),"threshold":float(threshold),
                      "f1":float(f),"precision":float(precision_score(y,pred,zero_division=0)),
                      "recall":float(recall_score(y,pred,zero_division=0))}
    return best

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pairs",required=True); args=ap.parse_args()
    df=pd.read_csv(args.pairs)
    missing={"text_sim","image_sim","label"}-set(df.columns)
    if missing: raise ValueError(f"missing columns: {sorted(missing)}")
    print(tune(df))

if __name__=="__main__":
    main()
