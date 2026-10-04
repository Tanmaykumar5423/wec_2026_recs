"""Part B: character TF-IDF retrieval baseline.
For full evaluation, generate labeled pairs and apply a validation threshold.
"""
from __future__ import annotations
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--csv",required=True); ap.add_argument("--query-index",type=int,default=0); args=ap.parse_args()
    df=pd.read_csv(args.csv)
    titles=df["title"].fillna("").astype(str)
    vectorizer=TfidfVectorizer(analyzer="char",ngram_range=(3,5),min_df=2,sublinear_tf=True,norm="l2")
    X=vectorizer.fit_transform(titles)
    q=max(0,min(args.query_index,len(df)-1))
    scores=cosine_similarity(X[q],X).ravel()
    idx=scores.argsort()[-10:][::-1]
    cols=[c for c in ["posting_id","title","label_group"] if c in df.columns]
    print(df.iloc[idx][cols].assign(similarity=scores[idx]).to_string(index=False))

if __name__=="__main__":
    main()
