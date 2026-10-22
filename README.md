# Classical Multidimensional Scaling (MDS): Theory, Implementation & Applications

> A from-scratch Python implementation of **classical (Torgerson) MDS**, compared against metric MDS on four real-world case studies: geography, word embeddings, image similarity and scientific tabular data.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)
![gensim](https://img.shields.io/badge/gensim-word2vec-green)

Undergraduate research project (Computer Science, Universidad Nacional de Colombia).

## Overview

Multidimensional Scaling recovers a low-dimensional configuration of points from a matrix of pairwise dissimilarities. That makes it a dimensionality-reduction tool for visualisation and pattern recognition when you only know *how different* objects are, not their coordinates.

This project:

- Derives the **mathematical foundations** of classical MDS: squared distances → double centring → Gram matrix → eigendecomposition.
- **Implements it from scratch** (`classic_MDS(D, k)`), including a non-matrix version of the Gram matrix computation.
- **Validates** the implementation against `sklearn.manifold.MDS` (metric, SMACOF) using Kruskal's **stress** and visual comparison.
- Applies both methods to **metric and non-metric data** to show where classical MDS works well and where it struggles.

## Case studies

| # | Case | Dissimilarity | What it shows |
|---|---|---|---|
| 1 | **Colombian cities** (`co.csv`) | Haversine great-circle distance | Classical MDS recovers the map of Colombia almost exactly from distances alone (stress ≈ 0.0003) |
| 2 | **Words in the plane**: Brown corpus | Co-occurrence matrix vs. Word2Vec embeddings | Synonyms and antonyms of *"president"*; top-positive vs. top-negative words in 2-D and 3-D |
| 3 | **Image similarity**: USC-SIPI test images | Gaussian noise, text watermarks, Haar/Daubechies wavelet transforms; Hausdorff distance | Groups distorted versions of the same image |
| 4 | **Scientific tabular data** | Correlation and Mahalanobis distance | Structure in multivariate data; comparison with metric MDS in k = 2 and k = 3 |

Interactive Plotly outputs (open in a browser):
[`ciudades_col_interactivo.html`](ciudades_col_interactivo.html) · [`ciudades_col_reducido_interactivo.html`](ciudades_col_reducido_interactivo.html) · [`ciudades_col_reducido_3d_interactivo.html`](ciudades_col_reducido_3d_interactivo.html)

## Core algorithm

```python
def classic_MDS(D, k=2):
    D_2 = np.square(D)                                   # squared distances
    # Gram matrix via double centring (element-wise, no centring matrix)
    G = -0.5 * (D_2 - D_2.mean(axis=1, keepdims=True)
                    - D_2.mean(axis=0, keepdims=True) + D_2.mean())
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    idx = np.argsort(eigenvalues)[::-1]                  # largest first
    L = np.diag(np.sqrt(eigenvalues[idx][:k]))
    V = eigenvectors[:, idx][:, :k]
    return V @ L                                          # k-dimensional embedding
```

## Repository contents

```
├── MultidimensionaScaling_LNBS.ipynb   # main notebook: theory, implementation, case studies
├── co.csv                              # Colombian cities with coordinates
├── brown.embedding                     # word embedding file
├── Img_clasificacion/                  # USC-SIPI test images
├── ciudades_col_*.html                 # interactive Plotly visualisations
├── Material_respaldo/                  # supporting material and references
└── requirements.txt
```

## How to run

```bash
git clone https://github.com/bermudez05/MDS_clasico.git
cd MDS_clasico
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook MultidimensionaScaling_LNBS.ipynb
```

## Tech stack

Python · NumPy · SciPy · scikit-learn · gensim · NLTK · Pillow · PyWavelets · scikit-image · Matplotlib · Plotly

## Author

**Laura Nicole Bermúdez Santa**, Universidad Nacional de Colombia · [GitHub](https://github.com/bermudez05)
