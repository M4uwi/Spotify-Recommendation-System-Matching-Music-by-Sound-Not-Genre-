import os
import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

def load_config(config_path="config.yaml"):
    """Load hyperparameters and dataset paths from configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}.")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_model():
    # 1. Load Configurations
    config = load_config()
    data_path = config["data"].get("dataset_url")
    max_samples = config["data"].get("max_samples", 25000)
    
    print(f"Loading data from source: {data_path}...")
    df = pd.read_csv(data_path)
    
    # Preprocessing & Deduplication
    df = df.drop_duplicates(subset=['track_name', 'track_artist']).reset_index(drop=True)
    df = df.dropna().head(max_samples)
    
    acoustic_features = config["features"]["acoustic_attributes"]
    X = df[acoustic_features].values
    
    print(f"Extracted {X.shape[1]} acoustic features across {X.shape[0]} tracks.")
    
    # 2. Build Unsupervised Pipeline (Scaler + KMeans)
    n_clusters = config["model"].get("n_clusters", 8)
    random_state = config["model"].get("random_state", 42)
    
    print(f"Building unsupervised feature processing pipeline (KMeans clusters: {n_clusters})...")
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(scaled_features)
    df['cluster'] = clusters
    
    # 3. PCA Dimensionality Reduction
    pca_components = config["model"].get("pca_components", 2)
    print(f"Computing PCA projection (Components: {pca_components})...")
    
    pca = PCA(n_components=pca_components, random_state=random_state)
    pca_coords = pca.fit_transform(scaled_features)
    df['pca1'] = pca_coords[:, 0]
    df['pca2'] = pca_coords[:, 1]
    
    # 4. Compute Cosine Similarity Matrix
    print("Computing multi-variable Cosine Similarity matrix...")
    cosine_sim_matrix = cosine_similarity(scaled_features, scaled_features)
    
    # 5. Pipeline Validation Metrics
    print("\n" + "="*30)
    print("--- Pipeline Execution Summary ---")
    print(f"Processed Tracks: {len(df)}")
    print(f"Acoustic Dimensions: {len(acoustic_features)}")
    print(f"Feature Space Bound: [{scaled_features.min():.1f}, {scaled_features.max():.1f}]")
    print(f"Similarity Matrix Shape: {cosine_sim_matrix.shape}")
    print("="*30 + "\n")
    
    # 6. Artifact Persistence
    artifact_dir = "models/artifacts"
    os.makedirs(artifact_dir, exist_ok=True)
    
    processed_data_path = os.path.join(artifact_dir, "processed_spotify_data.joblib")
    sim_matrix_path = os.path.join(artifact_dir, "cosine_similarity_matrix.joblib")
    scaler_path = os.path.join(artifact_dir, "minmax_scaler.joblib")
    
    print(f"Saving serialized artifacts to {artifact_dir}...")
    joblib.dump(df, processed_data_path)
    joblib.dump(cosine_sim_matrix, sim_matrix_path)
    joblib.dump(scaler, scaler_path)
    
    print("Training complete! Model artifacts saved and ready for production dashboard.")

if __name__ == "__main__":
    train_model()
