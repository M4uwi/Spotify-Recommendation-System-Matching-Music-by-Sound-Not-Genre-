# Matching Music by Sound, Not Genre
### A Content-Based Spotify Recommendation Engine

* **Author:** Kenneth Kinungi Mauwi
* **Department:** Department of Electrical Engineering, JKUAT
* **Program:** Wecncode DS/ML Bootcamp

---

## Project Overview
Current music streaming recommendation algorithms heavily rely on popularity bias and strict categorization, trapping users in "genre bubbles" and severely limiting organic discovery. Matching Music by Sound, Not Genre addresses this challenge by utilizing Content-Based Filtering. Instead of depending on human-assigned genre tags or user listening history, the engine matches tracks purely on fundamental mathematical attributes—analyzing a song's structural "acoustic DNA".

---

## Similarity Index Model and Algorithm
The core engine computes similarity scores across continuous feature vectors to measure the proximity between tracks in a high-dimensional feature space.

### 1. Acoustic DNA Representation
Each track $i$ is transformed into an $N$-dimensional vector $X_i$:

$$X_i = [f_1, f_2, \dots, f_{11}]$$

where $f_1, \dots, f_{11}$ correspond to the 11 continuous acoustic attributes (`danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, and `tempo`).

### 2. Feature Normalization (MinMaxScaler)
To prevent features with larger scales (such as tempo or loudness in decibels) from dominating distance calculations, all variables are normalized to a uniform scale of $[0, 1]$:

$$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

### 3. Cosine Similarity Index
The mathematical similarity between a user-selected reference track vector $A$ and a candidate dataset track vector $B$ is measured using Cosine Similarity. Cosine similarity calculates the cosine of the angle between the two vectors in multi-dimensional space:

$$\text{Similarity}(A, B) = \cos(\theta) = \frac{A \cdot B}{\Vert{}A\Vert{} \Vert{}B\Vert{}} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

* **Range:** Outputs a bounded score between $0$ (completely orthogonal/dissimilar) and $1$ (identical directional profile).
* **Properties:** By evaluating the angle rather than magnitude alone, the metric captures structural harmony across relative audio attributes.

### 4. Ranking and Retrieval
Upon input of a reference track, the algorithm computes pairwise similarity against all vectors in the dataset, sorts the array in descending order, and retrieves the top $N$ closest matches (excluding the input track itself).

---

## Dataset Details and EDA Summary
* **Source:** Spotify Dataset (Kaggle)
* **Scale:** Over 32,000 unique tracks processed and cleaned
* **Acoustic Attributes:** Mapped across 11 continuous acoustic dimensions
* **Exploratory Data Analysis:** Visualized feature distributions and correlation matrices to analyze popularity score skews and genre track counts (covering Pop, Rap, EDM, Rock, Latin, and R&B)

---

## Model Evaluation and Validation Metrics
Because content-based nearest-neighbor systems operate without a direct classification label, system performance and integrity are validated through:

1. **Feature Scaling Verification:** Applied `MinMaxScaler` across all continuous dimensions to prevent feature dominance during distance computation.
2. **Intra-List Similarity (ILS):** Evaluated diversity among top recommendations to confirm the engine surfaces varied structural tracks rather than redundant exact matches.
3. **K-Means Silhouette Analysis:** Measured cluster cohesion and separation for acoustic mood partitioning, verifying distinct spatial boundaries.
4. **Geometric Validation:** Cross-referenced multi-variable similarity outputs against manual radar plot overlays.

---

## Dashboard Details
* **Framework:** Python, Gradio / Dash, Plotly, Pandas
* **Search Control Panel:** Dynamic track lookup by title and artist
* **Recommendation Table:** Displays top mathematically similar tracks along with rank, artist, genre, popularity score, and match percentage
* **Visualization Panel:** Polar radar charts and PCA spatial maps for visual validation

---

## Broader Business Applications
While implemented for music discovery, this similarity model applies directly to corporate data challenges, such as in the Kenyan fintech and banking sector:
* **The Challenge:** Major financial institutions often face discovery bottlenecks where recommendation algorithms suffer from popularity bias, pushing generic products while burying specialized plans.
* **The Solution:** By translating this similarity framework from "acoustic DNA" to "transactional DNA," institutions can match clients to tailored financial products based on behavioral feature vectors.

---

## Repository Structure
```text
spotify-recommendation-engine/
│
├── config.yaml
├── README.md
│
├── modelling/
│   ├── train_model.py
│   
└── dashboard/
    ├── app.py
