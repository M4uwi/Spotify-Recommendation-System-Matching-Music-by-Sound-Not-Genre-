import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# 1. PAGE CONFIGURATION & SPOTIFY PREMIUM DARK THEME
# ==============================================================================
st.set_page_config(
    page_title="Spotify Professional Recommendation Engine",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Dark Palette (Charcoal, Deep Emerald Accents, Sleek Grays)
BG_PRIMARY = "#0e1117"
BG_CARD = "#161b22"
ACCENT_GREEN = "#2ea043"
ACCENT_CYAN = "#38bdf8"
TEXT_MAIN = "#f0f6fc"
TEXT_SUB = "#8b949e"
BORDER_COLOR = "#30363d"

st.markdown(f"""
    <style>
    .main {{
        background-color: {BG_PRIMARY};
        color: {TEXT_MAIN};
    }}
    .stSelectbox label {{
        color: {TEXT_MAIN} !important;
        font-weight: 600;
        font-size: 16px;
    }}
    div.stButton > button {{
        background: linear-gradient(135deg, {ACCENT_GREEN}, #238636);
        color: white;
        border: none;
        font-weight: bold;
        border-radius: 6px;
        transition: 0.3s ease;
    }}
    div.stButton > button:hover {{
        background: linear-gradient(135deg, #3fb950, {ACCENT_GREEN});
        border-color: #3fb950;
    }}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. LOAD & CACHE DATASET & MODEL MATRIX
# ==============================================================================
@st.cache_data
def load_and_process_data():
    df = pd.read_csv('spotify_songs.csv')
    df = df.drop_duplicates(subset=['track_id']).dropna(subset=['track_name', 'track_artist']).reset_index(drop=True)
    df['search_label'] = df['track_name'] + ' — ' + df['track_artist']
    
    # Use strictly acoustic/structural audio features for true cross-genre sonic matching
    audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                      'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    
    scaler = MinMaxScaler()
    # Feature matrix consists entirely of normalized audio DNA—no genre bias
    feature_matrix = scaler.fit_transform(df[audio_features])
    
    return df, feature_matrix, audio_features

df, FINAL_FEATURE_MATRIX, AUDIO_FEATURES = load_and_process_data()

# ==============================================================================
# 3. HEADER SECTION
# ==============================================================================
st.title("🎧 Spotify Professional Recommendation Engine")
st.markdown(f"<p style='color: {TEXT_SUB}; font-size: 16px;'>Select a reference track below to analyze acoustic profiles and generate the top 20 mathematically similar songs using multi-variable cosine similarity.</p>", unsafe_allow_html=True)
st.markdown("---")

# ==============================================================================
# 4. TOP SECTION: SEARCH BAR & REFERENCE CONTROL
# ==============================================================================
col1, col2 = st.columns([3, 1])

with col1:
    song_options = df['search_label'].unique().tolist()
    selected_song = st.selectbox(
        "🔍 Search for a Reference Song",
        options=song_options,
        index=0
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_clicked = st.button("Find Top 20 Recommendations", use_container_width=True)

# ==============================================================================
# 5. CORE COMPUTATION (COSINE SIMILARITY - TOP 20)
# ==============================================================================
ref_idx = df[df['search_label'] == selected_song].index[0]
ref_row = df.iloc[ref_idx]

target_vector = FINAL_FEATURE_MATRIX[ref_idx].reshape(1, -1)
sim_scores = cosine_similarity(target_vector, FINAL_FEATURE_MATRIX).flatten()
# Fetch top 20 recommendations (excluding index 0 which is the track itself)
top_indices = sim_scores.argsort()[-21:-1][::-1]

# ==============================================================================
# 6. MIDDLE SECTION: TOP 20 RECOMMENDATIONS TABLE
# ==============================================================================
st.markdown(f"### 🎵 Top 20 Recommended Tracks for *{ref_row['track_name']}*")

table_data = []
for rank, idx in enumerate(top_indices, start=1):
    row = df.iloc[idx]
    score_pct = f"{sim_scores[idx] * 100:.2f}%"
    table_data.append({
        'Rank': rank,
        'Track Name': row['track_name'],
        'Artist': row['track_artist'],
        'Genre': row['playlist_genre'].title(),
        'Popularity': row['track_popularity'],
        'Similarity Match': score_pct
    })

df_recommendations = pd.DataFrame(table_data)

st.dataframe(
    df_recommendations,
    use_container_width=True,
    hide_index=True,
    height=400
)

st.markdown("---")

# ==============================================================================
# 7. BOTTOM SECTION: AUDIO FEATURE COMPARISON RADAR CHART
# ==============================================================================
st.markdown("### 📊 Audio Feature Comparison (Sonic Footprint Radar)")
st.markdown(f"<p style='color: {TEXT_SUB}; font-size: 14px;'>Visual comparison of normalized audio DNA metrics between your reference track and the #1 recommended match.</p>", unsafe_allow_html=True)

top_match_idx = top_indices[0]
top_match_row = df.iloc[top_match_idx]

radar_scaler = MinMaxScaler()
scaled_radar_vals = radar_scaler.fit_transform(df[AUDIO_FEATURES])

ref_vals = scaled_radar_vals[ref_idx]
match_vals = scaled_radar_vals[top_match_idx]

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=ref_vals,
    theta=[feat.capitalize() for feat in AUDIO_FEATURES],
    fill='toself',
    name=f"Reference: {ref_row['track_name'][:20]}...",
    line=dict(color=ACCENT_CYAN, width=2.5),
    fillcolor='rgba(56, 189, 248, 0.2)'
))

fig.add_trace(go.Scatterpolar(
    r=match_vals,
    theta=[feat.capitalize() for feat in AUDIO_FEATURES],
    fill='toself',
    name=f"Match #1: {top_match_row['track_name'][:20]}...",
    line=dict(color=ACCENT_GREEN, width=2.5),
    fillcolor='rgba(46, 160, 67, 0.2)'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1], color=TEXT_SUB, gridcolor=BORDER_COLOR),
        angularaxis=dict(color=TEXT_MAIN, tickfont=dict(size=12)),
        bgcolor=BG_CARD
    ),
    paper_bgcolor=BG_PRIMARY,
    plot_bgcolor=BG_PRIMARY,
    font=dict(color=TEXT_MAIN),
    legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
    margin=dict(l=50, r=50, t=60, b=40),
    height=520
)

st.plotly_chart(fig, use_container_width=True)