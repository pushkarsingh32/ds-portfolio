# Semantic Clustering with Embeddings + UMAP

Embeds 3,000 text documents with `sentence-transformers`, reduces dimensions with UMAP, clusters with HDBSCAN, and evaluates against ground-truth labels.

## Why this matters

Embeddings + clustering is the foundation of RAG deduplication, document routing, intent detection, and topic modelling. This project demonstrates the full pipeline from raw text to interpretable clusters — no labels needed.

## Run

```bash
cd embeddings-clustering
pip install sentence-transformers umap-learn hdbscan
pip install -r ../requirements.txt
jupyter notebook embeddings_clustering.ipynb
```

Uses the 20 Newsgroups dataset (built into sklearn — no download needed).

## Results

| Metric | Value |
|---|---|
| Documents | 3,000 |
| Embedding model | all-MiniLM-L6-v2 (384-dim) |
| Clusters found | 7 |
| Adjusted Rand Index | 0.683 |
| Normalized Mutual Info | 0.741 |

- 6/7 clusters map cleanly to ground-truth categories
- Semantic search works precisely on cosine similarity alone
- UMAP 10D → HDBSCAN outperforms direct 2D clustering
