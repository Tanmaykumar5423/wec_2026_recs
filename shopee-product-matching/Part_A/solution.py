"""Part A: lightweight Shopee EDA.
Expected columns commonly include posting_id, image, title and label_group.
"""
from __future__ import annotations
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--csv",required=True); args=ap.parse_args()
    df=pd.read_csv(args.csv)
    print("shape:",df.shape)
    print("columns:",list(df.columns))
    print("missing-rate:")
    print(df.isna().mean().sort_values(ascending=False).head(20))
    if "label_group" in df:
        gs=df.groupby("label_group").size()
        print("unique groups:",gs.size)
        print(gs.describe())
        gs.clip(upper=20).value_counts().sort_index().plot.bar()
        plt.title("Product-group size distribution")
        plt.xlabel("group size, clipped at 20")
        plt.ylabel("number of groups")
        plt.tight_layout(); plt.show()
    if "title" in df:
        length=df["title"].fillna("").astype(str).str.len()
        print("title length:")
        print(length.describe())
        length.plot.hist(bins=40)
        plt.title("Title character length")
        plt.xlabel("characters")
        plt.tight_layout(); plt.show()
    if "image" in df:
        print("unique image references:",df["image"].nunique())
    if "title" in df and "label_group" in df:
        dup=df.groupby("title").label_group.nunique()
        print("exact titles mapping to multiple groups:",int((dup>1).sum()))

if __name__=="__main__":
    main()
