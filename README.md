# Topic Modeling Using Latent Dirichlet Allocation, Latent Semantic Analysis, and BERT Transformer & Its Applications

## Overview

This project explores the use of three powerful techniques for topic modeling:

1. **Latent Dirichlet Allocation (LDA)** - A generative probabilistic model leveraging the Dirichlet and multinomial distributions.
2. **Latent Semantic Analysis (LSA)** - A dimensionality reduction approach based on singular value decomposition (SVD).
3. **BERTopic Pipeline** - A modern topic modeling approach utilizing BERT embeddings and clustering techniques.

## Datasets

We used three datasets to evaluate the topic modeling methods:

1. [Amazon US Customer Reviews Dataset](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset/data)
2. [Global News Dataset](https://www.kaggle.com/datasets/everydaycodings/global-news-dataset)
3. [Lenovo K8 Review Dataset](https://www.kaggle.com/datasets/abhiram8/lenovok8review)

## Learning Outcomes

Throughout the project, we gained insights into:

- **Dirichlet and Multinomial Distributions**: Fundamental components of LDA.
- **Generative Probabilistic Models**: How LDA generates topics and documents.
- **Singular-Value Decomposition (SVD)**: Used for dimensionality reduction in LSA.
- **BERT Embeddings**: Capturing semantic relationships between words and documents.
- **UMAP for Dimensionality Reduction**: Effective visualization and preprocessing for clustering.
- **Clustering Techniques**:
  - **k-means clustering**
  - **HDBSCAN clustering**
- **Topic Modeling Evaluation Metrics**:
  - **Coherence Score (c o)**: Measuring the interpretability of topics.
  - **Topic Diversity (t d)**: Assessing topic representation across documents.

## Results

The following tables summarize the evaluation metrics for LDA, LSA, and BERTopic across the three datasets:

### Table 1: Scores for LSA across datasets

| Metric | Dataset I | Dataset II | Dataset III |
| ------ | --------- | ---------- | ----------- |
| c o    | 0.5158    | 0.8909     | 0.4661      |
| t d    | 0.7333    | 0.7333     | 0.5714      |

### Table 2: Scores for BERTopic across datasets

| Metric | Dataset I | Dataset II | Dataset III |
| ------ | --------- | ---------- | ----------- |
| c o    | 0.4729    | 0.6242     | 0.4534      |
| t d    | 0.5142    | 0.8662     | 0.6000      |

### Table 3: Scores for LDA across datasets

| Metric | Dataset I | Dataset II | Dataset III |
| ------ | --------- | ---------- | ----------- |
| c o    | 0.4428    | 0.4208     | 0.6117      |
| t d    | 0.6972    | 0.8142     | 0.7862      |

## Streamlit App

An interactive Streamlit app was built to visualize and explore the results of topic modeling. The app provides the following functionalities:

1. Visualize barcharts of top words per topic with adjustable topic count.
2. Display topic-topic similarity as a heatmap.
3. Explore topic hierarchies using a dendrogram.
4. Visualize topic distributions with interactive plots.
5. Generate word clouds for individual topics.
6. Analyze a news article to compute and display topic probabilities.
7. Perform interactive filtering and keyword-based topic search.
8. Examine the importance of keywords within topics.
9. Find similar articles based on content similarity.
10. Dynamically add new documents to update the topic model.

---

<!-- Stay tuned for more updates and detailed discussions in this repository! -->

## This project was completed as a part of capstone project for **Samsung Innovation Campus AI programme**. For more information go through the report uploaded.
