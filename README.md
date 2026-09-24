# 🎧 Matching Music by Sound, Not Genre
### A Content-Based Spotify Recommendation Engine

## Description
Traditional music recommendation systems heavily rely on popularity bias and rigid categorical genre tags, trapping users in "genre bubbles" and burying underground or cross-genre tracks. **Matching Music by Sound, Not Genre** solves this limitation by implementing a content-based filtering engine. Instead of looking at user-history or mainstream labels, the system analyzes a song's fundamental "acoustic DNA" to discover and recommend structurally similar tracks across boundaries.

---

## Features
* **Acoustic-First Matching:** Computes multi-variable cosine similarity across normalized audio features to pair tracks based on rhythm, energy, and mood rather than human-assigned genre labels.
* **Interactive Dash Web Application:** Professional dark-themed UI styled after Spotify’s design system, featuring a search dropdown, data table, and real-time metric updates.
* **Sonic Footprint Radar Charts:** Interactive Plotly polar charts that visually compare the normalized audio attributes of a reference track against its top matches.
* **Advanced Spatial Mapping & Clustering:** Integrates Principal Component Analysis (PCA) for 2D acoustic visualization and K-Means clustering to partition audio space into distinct mood neighborhoods.

---

## Dataset Details
* **Source:** Spotify Songs Dataset (Kaggle).
* **Scale:** Over 32,000 unique tracks after cleaning and removing duplicate identifiers.
* **Core Acoustic Attributes (11 Features):** `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, and `tempo`.
* **Preprocessing:** Handled null values, dropped redundant track duplicates, and normalized all continuous numerical variables using scikit-learn's `MinMaxScaler`.

---

## Algorithm Details
* **Content-Based Filtering:** Recommends items similar to those a user likes, based purely on item feature metadata.
* **Cosine Similarity:** Measures the cosine of the angle between multi-dimensional vectors, quantifying how closely aligned two tracks' acoustic profiles are in vector space.
* **K-Means Clustering:** Partitions the continuous audio space into optimal spherical clusters to group tracks by shared acoustic moods.
* **Principal Component Analysis (PCA):** Compresses high-dimensional 11-feature vectors down to 2 principal components for variance-preserving spatial mapping.

---

## Dashboard Details
* **Framework:** Built using Python, Dash, Plotly, and Pandas.
* **Layout Structure:** 
  * *Control Panel:* Searchable track dropdown and recommendation trigger button.
  * *Visualization Panel:* Dynamic Plotly polar radar charts tracking geometric audio feature alignment.
  * *Data Table:* Formatted table displaying the top mathematically similar tracks, complete with rank, artist, genre, popularity score, and similarity match percentage.
* **Color Palette:** Professional Spotify aesthetic.

---

## Insights from the Dataset
* **Popularity Skew:** Exploratory data analysis revealed that track popularity scores heavily skew toward lower indices, highlighting a steep drop-off between mainstream chart hits and independent catalog items.
* **Genre Boundaries vs. Acoustic Overlap:** Dropping categorical genre dummy variables proved that tracks across entirely different categories (e.g., matching a high-energy Rock track with an equivalent high-energy EDM track) often share identical structural audio DNA.
* **Feature Independence:** Multi-variable correlation analysis demonstrated that features like energy and loudness share strong positive relationships, while acousticness and energy show inverse scaling trends.

---

## Future Improvements
* **Audio Feature Expansion:** Integrate deeper audio embeddings derived directly from raw audio file analysis (e.g., using Librosa or pre-trained neural audio models).
* **Vector Database Integration:** Scale the nearest-neighbor backend using specialized vector search engines (like FAISS or Pinecone) for instantaneous retrieval across millions of tracks.
* **User Profile Customization:** Add user session tracking to aggregate multi-song preference vectors for dynamic playlist generation.
