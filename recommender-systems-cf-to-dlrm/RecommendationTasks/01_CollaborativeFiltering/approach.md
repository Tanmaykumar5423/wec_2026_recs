# Approach — Collaborative Filtering

## Strategy
Use MovieLens or another explicit user-item rating dataset.

1. Validate user, item and rating columns.
2. Map IDs to contiguous indices.
3. Use a fixed train/test split.
4. Build item-item cosine similarity as the main memory-based baseline.
5. Implement regularized matrix factorization with SGD.
6. Compare RMSE and MAE. For recommendation quality, add Precision@K and Recall@K using held-out interactions.
7. Run controlled sweeps over neighbor count, latent dimension and regularization.

## Matrix factorization
Approximate R by P Q transpose and predict:
r_hat(u,i) = global_mean + user_bias + item_bias + P_u dot Q_i.

Optimize squared error with L2 regularization.

## What to explain
Discuss sparsity, cold start, interpretability, computational cost, popularity effects and why latent factors can generalize better than raw neighborhoods.
