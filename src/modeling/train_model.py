# ==========================================
# MODELLING & PIPELINE SCRIPT
# ==========================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def run_pipeline():
    # 1. Ingest Data
    DATASET_URL = "https://raw.githubusercontent.com/mafgl/Spotify-Dataset/master/spotify_songs.csv"
    df = pd.read_csv(DATASET_URL)
    
    # Clean data
    df = df.drop_duplicates(subset=['track_name', 'track_artist']).reset_index(drop=True)
    df = df.dropna().head(5000)
    
    # 2. Define Features
    ACOUSTIC_FEATURES = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 
        'speechiness', 'acousticness', 'instrumentalness', 
        'liveness', 'valence', 'tempo'
    ]
    X = df[ACOUSTIC_FEATURES].values
    
    # 3. Scale Features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(X)
    
    # 4. Compute Cosine Similarity Matrix
    cosine_sim = cosine_similarity(scaled_features, scaled_features)
    
    # 5. PCA & K-Means
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(scaled_features)
    df['pca1'] = pca_coords[:, 0]
    df['pca2'] = pca_coords[:, 1]
    
    kmeans = KMeans(n_clusters=8, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_features)
    
    print("Modelling pipeline executed successfully.")
    return df, cosine_sim

if __name__ == "__main__":
    run_pipeline()