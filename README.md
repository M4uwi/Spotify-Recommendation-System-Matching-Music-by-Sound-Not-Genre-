# Matching Music by Sound, Not Genre
### A Content-Based Spotify Recommendation Engine

---

## 📌 Project Overview
Current music streaming recommendation algorithms heavily rely on popularity bias and strict categorization, trapping users in "genre bubbles" and severely limiting organic discovery. **Matching Music by Sound, Not Genre** addresses this challenge by utilizing **Content-Based Filtering**. Instead of depending on human-assigned genre tags or user history, the engine matches tracks purely on fundamental mathematical attributes—analyzing a song's structural "acoustic DNA".

---

## 📊 Dataset Details & EDA Summary
* **Source:** Spotify Dataset (Kaggle).
* **Scale:** Over **32,000+ unique tracks** processed and cleaned.
* **Acoustic Attributes:** Mapped across **11 continuous acoustic dimensions**, including energy, tempo, valence, danceability, acousticness, and instrumentalness.
* **Exploratory Data Analysis:** Visualized feature distributions and correlation matrices to analyze popularity score skews and genre track counts (covering Pop, Rap, EDM, Rock, Latin, and R&B).

---

## ⚙️ Methodology & Algorithm Details
* **Content-Based Filtering:** Recommends tracks based on item similarity derived from internal feature profiles rather than collaborative user filtering.
* **Feature Scaling:** Employs `MinMaxScaler` to normalize continuous audio variables into a uniform range.
* **Cosine Similarity:** Computes multi-variable vector distance to measure the precise acoustic alignment between a reference track and the rest of the database.

---

## 💻 Dashboard & Interactive Demo
Built using Python, Dash, and Plotly, the web interface provides:
* **Search Control Panel:** Dynamic track lookup by name and artist.
* **Top 20 Recommendation Table:** Instantly displays mathematically similar tracks complete with similarity match percentages.
* **Sonic Footprint Radar Charts:** Polar plots that visually validate audio feature alignment between a reference song and its top matches.

---

## 📈 Broader Business & Enterprise Applications
While designed for music discovery, this underlying content-based similarity architecture addresses broader cross-industry challenges—such as in the Kenyan fintech and banking sector:
* **The Challenge:** Major institutions (like Safaricom or commercial banks) face discovery hurdles where recommendation algorithms suffer from popularity bias, aggressively pushing standard micro-loans while burying specialized savings or investment options.
* **The Solution:** By translating this similarity architecture from a song's "acoustic DNA" to a customer's "transactional DNA," enterprises can match clients to bespoke financial products based on behavioral metrics.

---

* **Author:** Kenneth Kinungi Mauwi
