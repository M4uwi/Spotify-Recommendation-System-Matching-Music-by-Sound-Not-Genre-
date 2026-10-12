# ==========================================
# DASHBOARD APPLICATION SCRIPT
# ==========================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# Load and prepare data for the interface
DATASET_URL = "https://github.com/M4uwi/Spotify-Recommendation-System-Matching-Music-by-Sound-Not-Genre-/blob/main/Dataset/spotify_songs.csv"
df = pd.read_csv(DATASET_URL)
df = df.drop_duplicates(subset=['track_name', 'track_artist']).reset_index(drop=True)

ACOUSTIC_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo'
]

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[ACOUSTIC_FEATURES].values)
cosine_sim = cosine_similarity(scaled_features, scaled_features)

def recommend_tracks(song_title: str):
    match = df[df['track_name'].str.lower() == song_title.lower()]
    if match.empty:
        return f"Error: Song '{song_title}' not found in the database."
    
    track_index = match.index[0]
    similarity_scores = list(enumerate(cosine_sim[track_index]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:21]
    
    recommended_indices = [item[0] for item in sorted_scores]
    match_percentages = [f"{item[1] * 100:.2f}%" for item in sorted_scores]
    
    results_df = df.iloc[recommended_indices][['track_name', 'track_artist', 'playlist_genre', 'popularity']].copy()
    results_df['Similarity Match'] = match_percentages
    results_df.insert(0, 'Rank', range(1, 21))
    
    return results_df

# Build Gradio UI
track_options = df['track_name'].dropna().unique().tolist()

with gr.Blocks() as demo:
    gr.Markdown("# 🎧 Matching Music by Sound, Not Genre")
    gr.Markdown("### Content-Based Spotify Recommendation Engine")
    
    with gr.Row():
        song_dropdown = gr.Dropdown(
            choices=track_options, 
            label="Search Reference Track", 
            value=track_options[0] if track_options else ""
        )
        submit_button = gr.Button("Find Top 20 Recommendations", variant="primary")
    
    output_dataframe = gr.Dataframe(label="Mathematically Similar Recommendations")
    
    submit_button.click(
        fn=recommend_tracks, 
        inputs=song_dropdown, 
        outputs=output_dataframe
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
