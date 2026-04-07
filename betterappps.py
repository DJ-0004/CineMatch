import streamlit as st
import requests
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import random
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch · Hybrid Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLES  (cinematic dark, amber accents)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #0a0a0f;
    color: #e8e0d0;
}

.stApp { background: #0a0a0f; }

/* hide default streamlit chrome */
#MainMenu, footer { visibility: hidden; }

/* ── HERO ── */
.hero {
    text-align: center;
    padding: 3.5rem 1rem 2rem;
    background: radial-gradient(ellipse 80% 50% at 50% -10%, rgba(255,190,60,.18) 0%, transparent 70%);
    border-bottom: 1px solid rgba(255,190,60,.12);
    margin-bottom: 2.5rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.4rem, 6vw, 4.2rem);
    font-weight: 900;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #ffd77a 0%, #ff9a3c 60%, #ff5e62 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 .5rem;
    line-height: 1.1;
}
.hero p {
    color: #8a8070;
    font-size: 1.05rem;
    font-weight: 300;
    margin: 0;
    letter-spacing: .4px;
}

/* ── SECTION LABEL ── */
.section-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: #ffd77a;
    border-left: 3px solid #ff9a3c;
    padding-left: .75rem;
    margin: 2.2rem 0 1.2rem;
}

/* ── MOVIE CARD ── */
.movie-card {
    background: linear-gradient(160deg, #14141e 0%, #0e0e16 100%);
    border: 1px solid rgba(255,190,60,.14);
    border-radius: 14px;
    overflow: hidden;
    transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
    height: 100%;
    display: flex;
    flex-direction: column;
    position: relative;
}
.movie-card:hover {
    transform: translateY(-5px) scale(1.012);
    box-shadow: 0 20px 50px rgba(255,154,60,.18);
    border-color: rgba(255,190,60,.45);
}
.card-poster-wrap {
    position: relative;
    width: 100%;
    padding-top: 148%;   /* 2:3 ratio */
    overflow: hidden;
    background: #0a0a12;
}
.card-poster-wrap img {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    object-fit: cover;
    transition: transform .4s ease;
}
.movie-card:hover .card-poster-wrap img {
    transform: scale(1.06);
}
.card-rank-badge {
    position: absolute;
    top: 10px; left: 10px;
    background: rgba(10,10,15,.85);
    border: 1px solid rgba(255,190,60,.5);
    color: #ffd77a;
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    font-size: .78rem;
    padding: 3px 8px;
    border-radius: 5px;
    backdrop-filter: blur(4px);
    z-index: 2;
}
.card-type-badge {
    position: absolute;
    top: 10px; right: 10px;
    font-size: .65rem;
    font-weight: 500;
    letter-spacing: .6px;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 5px;
    backdrop-filter: blur(4px);
    z-index: 2;
}
.badge-hybrid   { background: rgba(255,154,60,.25); color: #ff9a3c; border: 1px solid rgba(255,154,60,.4); }
.badge-content  { background: rgba(60,180,255,.20); color: #60c8ff; border: 1px solid rgba(60,180,255,.35); }
.badge-collab   { background: rgba(160,80,255,.20); color: #c07cff; border: 1px solid rgba(160,80,255,.35); }

.card-body { padding: 1rem 1rem .9rem; flex: 1; display: flex; flex-direction: column; gap: .55rem; }

.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.0rem;
    font-weight: 700;
    color: #f0e8d8;
    line-height: 1.25;
    margin: 0;
}
.card-meta {
    display: flex;
    align-items: center;
    gap: .5rem;
    font-size: .78rem;
    color: #6a6058;
}
.card-meta .year { color: #a09080; }
.card-meta .dot  { color: #3a3028; }
.card-meta .vote { color: #ffd77a; font-weight: 500; }

.card-overview {
    font-size: .78rem;
    color: #7a7060;
    line-height: 1.55;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    flex: 1;
}

/* score bars */
.score-row { display: flex; flex-direction: column; gap: 4px; margin-top: auto; }
.score-line { display: flex; align-items: center; gap: 6px; font-size: .72rem; }
.score-label { width: 62px; color: #6a6058; flex-shrink: 0; }
.score-bar-bg {
    flex: 1;
    height: 4px;
    background: rgba(255,255,255,.07);
    border-radius: 2px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width .6s ease;
}
.fill-hybrid  { background: linear-gradient(90deg, #ff9a3c, #ffd77a); }
.fill-content { background: linear-gradient(90deg, #3c9eff, #60c8ff); }
.fill-collab  { background: linear-gradient(90deg, #9a4cff, #c07cff); }
.score-val { width: 32px; text-align: right; color: #a09080; }

/* genre + person chips */
.chip-row { display: flex; flex-wrap: wrap; gap: 4px; }
.chip {
    font-size: .65rem;
    padding: 2px 7px;
    border-radius: 4px;
    font-weight: 500;
    letter-spacing: .3px;
}
.chip-genre  { background: rgba(255,190,60,.10); color: #c09030; border: 1px solid rgba(255,190,60,.2); }
.chip-person { background: rgba(255,255,255,.05); color: #8a8070; border: 1px solid rgba(255,255,255,.1); }

/* ── FILTER SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #0d0d16 !important;
    border-right: 1px solid rgba(255,190,60,.1);
}
[data-testid="stSidebar"] * { font-family: 'DM Sans', sans-serif !important; }

/* inputs */
.stTextInput input, .stNumberInput input {
    background: #14141e !important;
    border: 1px solid rgba(255,190,60,.2) !important;
    color: #e8e0d0 !important;
    border-radius: 8px !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #ff9a3c !important;
    box-shadow: 0 0 0 2px rgba(255,154,60,.15) !important;
}

/* multiselect */
.stMultiSelect [data-baseweb="tag"] {
    background: rgba(255,154,60,.2) !important;
    color: #ffd77a !important;
}

/* slider */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #ff9a3c !important;
}

/* button */
.stButton > button {
    background: linear-gradient(135deg, #ff9a3c 0%, #ffd77a 100%) !important;
    color: #0a0a0f !important;
    border: none !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: .95rem !important;
    border-radius: 9px !important;
    padding: .65rem 1.5rem !important;
    letter-spacing: .3px;
    transition: opacity .2s, transform .15s !important;
    width: 100%;
}
.stButton > button:hover { opacity: .88; transform: translateY(-1px) !important; }

/* spinner */
.stSpinner > div { border-top-color: #ff9a3c !important; }

/* metric */
[data-testid="metric-container"] {
    background: #14141e;
    border: 1px solid rgba(255,190,60,.15);
    border-radius: 10px;
    padding: .8rem 1rem;
}
[data-testid="stMetricValue"] { color: #ffd77a !important; font-family: 'Playfair Display', serif !important; font-size: 1.7rem !important; }
[data-testid="stMetricLabel"] { color: #6a6058 !important; font-size: .8rem !important; }

/* info / warning boxes */
.stAlert { border-radius: 10px !important; border-left-color: #ff9a3c !important; background: rgba(255,154,60,.08) !important; }

/* expander */
.streamlit-expanderHeader { color: #a09080 !important; font-size: .85rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
TMDB_BASE   = "https://api.themoviedb.org/3"
POSTER_BASE = "https://image.tmdb.org/t/p/w500"
NO_POSTER   = "https://via.placeholder.com/300x450/0a0a12/ffd77a?text=No+Poster"

GENRE_MAP = {
    28:"Action",12:"Adventure",16:"Animation",35:"Comedy",80:"Crime",
    99:"Documentary",18:"Drama",10751:"Family",14:"Fantasy",36:"History",
    27:"Horror",10402:"Music",9648:"Mystery",10749:"Romance",
    878:"Science Fiction",10770:"TV Movie",53:"Thriller",10752:"War",37:"Western"
}
GENRES_LIST = sorted(GENRE_MAP.values())

# ─────────────────────────────────────────────
# TMDB HELPERS
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def tmdb_discover(tmdb_key, genre_ids, page):
    # 1. Define the URL
    url = "https://api.themoviedb.org/3/discover/movie"
    
    # 2. Define the params (THIS is what your editor was saying was missing!)
    params = {
        "api_key": tmdb_key,
        "with_genres": genre_ids,
        "page": page
    }
    
    # 3. Print for debugging (optional but helpful)
    print(f"Attempting to connect to: {url}")
    
    # 4. Make the request safely
    try:
        r = requests.get(url, params=params, timeout=10) 
        r.raise_for_status() 
        return r.json()
        
    except requests.exceptions.ConnectionError:
        st.error("Network error: Unable to reach the TMDB database. Please check your internet connection.")
        return {} 
        
    except requests.exceptions.Timeout:
        st.warning("The request to TMDB timed out. The server might be slow right now.")
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def tmdb_movie_details(tmdb_key, movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": tmdb_key,"append_to_response": "videos"}
    
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
        
     
        
    except :
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def tmdb_search(api_key: str, query: str) -> list[dict]:
    r = requests.get(
        f"{TMDB_BASE}/search/movie",
        params={"api_key": api_key, "query": query, "page": 1},
        timeout=10,
    )
    if r.status_code != 200:
        return []
    return r.json().get("results", [])


def genre_ids_for_names(names: list[str]) -> list[int]:
    rev = {v: k for k, v in GENRE_MAP.items()}
    return [rev[n] for n in names if n in rev]


def poster_url(path: str | None) -> str:
    return f"{POSTER_BASE}{path}" if path else NO_POSTER


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────
def build_feature_string(movie: dict) -> str:
    """
    Combine genres, keywords, director, cast, and overview
    into a single weighted text feature for TF-IDF.
    """
    parts = []

    # genres (repeated for weight)
    for g in movie.get("genres", []):
        name = g.get("name", "").lower().replace(" ", "")
        parts += [name] * 3

    # keywords
    for kw in (movie.get("keywords", {}) or {}).get("keywords", [])[:12]:
        parts.append(kw.get("name", "").lower().replace(" ", "_"))

    # director (repeated)
    credits = movie.get("credits", {}) or {}
    for crew in credits.get("crew", []):
        if crew.get("job") == "Director":
            parts += [crew.get("name", "").lower().replace(" ", "")] * 2
            break

    # top cast
    for actor in credits.get("cast", [])[:6]:
        parts.append(actor.get("name", "").lower().replace(" ", ""))

    # overview bag-of-words (light)
    overview = movie.get("overview", "")
    if overview:
        parts.append(overview.lower())

    return " ".join(parts)


# ─────────────────────────────────────────────
# CONTENT-BASED FILTERING
# ─────────────────────────────────────────────
def content_scores(query_feature: str, corpus_features: list[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8000,
        sublinear_tf=True,
    )
    all_features = [query_feature] + corpus_features
    tfidf_matrix = vectorizer.fit_transform(all_features)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return scores


# ─────────────────────────────────────────────
# COLLABORATIVE FILTERING  (SVD / matrix factorisation)
# ─────────────────────────────────────────────
def build_synthetic_ratings(movies: list[dict], n_users: int = 120, seed: int = 42) -> pd.DataFrame:
    """
    Synthesise a user × movie rating matrix based on movie popularity &
    vote_average so the SVD has something meaningful to factorise.
    (In production you'd use real user logs.)
    """
    rng = np.random.default_rng(seed)
    movie_ids = [m["id"] for m in movies]
    pop_scores = np.array([m.get("popularity", 1) for m in movies], dtype=float)
    vote_avg   = np.array([m.get("vote_average", 5) for m in movies], dtype=float)

    # Normalise signals to [0,1]
    # Normalise signals to [0,1]
    pop_scores = (pop_scores - pop_scores.min()) / (np.ptp(pop_scores) + 1e-9)
    vote_norm  = (vote_avg  - vote_avg.min())   / (np.ptp(vote_avg) + 1e-9)
    base_prob  = 0.35 * pop_scores + 0.65 * vote_norm   # probability user rated

    ratings_dict: dict[tuple, float] = {}
    for u in range(n_users):
        # Each user rates a random subset of movies (sparse)
        mask = rng.random(len(movies)) < base_prob
        for j, rated in enumerate(mask):
            if rated:
                # Rating influenced by vote_average + noise
                r = vote_avg[j] * 0.7 + rng.uniform(0, 3)
                r = float(np.clip(r, 1, 10))
                ratings_dict[(u, movie_ids[j])] = r

    rows, cols, vals = [], [], []
    mid_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    for (u, mid), v in ratings_dict.items():
        rows.append(u)
        cols.append(mid_to_idx[mid])
        vals.append(v)

    mat = csr_matrix((vals, (rows, cols)), shape=(n_users, len(movies)))
    return pd.DataFrame.sparse.from_spmatrix(mat, columns=movie_ids)


def svd_scores(ratings_df: pd.DataFrame, movie_ids_query: list[int]) -> np.ndarray:
    """
    Run truncated SVD and compute a pseudo-score for each movie
    based on its latent-factor representation.
    """
    mat = ratings_df.values
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    mat = mat.astype(float)

    # Centre rows (user means)
    row_means = np.true_divide(mat.sum(1), (mat != 0).sum(1) + 1e-9)
    mat_c = mat.copy()
    non_zero = mat != 0
    mat_c[non_zero] -= row_means[np.where(non_zero)[0]]

    k = min(15, min(mat_c.shape) - 1)
    # Convert to a sparse matrix first
    sparse_mat_c = csr_matrix(mat_c)

    # Find the smallest dimension of your matrix
    min_dim = min(sparse_mat_c.shape)

    # Calculate a safe 'k' that obeys the mathematical rule
    safe_k = min(k, min_dim - 1)

    # Handle the extreme edge case (if the matrix is too small)
    if safe_k <= 0:
        return {} # Return an empty dict/list depending on what your app expects

    # Run the algorithm safely
    U, sigma, Vt = svds(sparse_mat_c, k=safe_k)
    # Item latent vectors → L2 norm as quality proxy
    item_latent = (Vt.T * sigma)       # shape: (n_movies, k)
    raw = np.linalg.norm(item_latent, axis=1)

    # Normalise to [0,1]
    if raw.max() > raw.min():
        return (raw - raw.min()) / (raw.max() - raw.min())
    return np.zeros_like(raw)


# ─────────────────────────────────────────────
# HYBRID SCORER
# ─────────────────────────────────────────────
def hybrid_recommend(
    movies: list[dict],
    query_movie: dict | None,
    genre_names: list[str],
    w_content: float = 0.55,
    w_collab:  float = 0.45,
    top_n: int = 9,
) -> pd.DataFrame:
    """
    Returns a DataFrame of top_n recommended movies with all score columns.
    """
    n = len(movies)
    if n == 0:
        return pd.DataFrame()

    # ── 1. Content scores ──────────────────────────────────────────
    corpus_features = [build_feature_string(m) for m in movies]

    if query_movie:
        q_feat = build_feature_string(query_movie)
    else:
        # Build a virtual "query" from selected genres
        q_feat = " ".join(g.lower().replace(" ", "") + " " * 3 for g in genre_names)

    c_scores = content_scores(q_feat, corpus_features)

    # ── 2. Collaborative scores ────────────────────────────────────
    ratings_df = build_synthetic_ratings(movies)
    cf_scores  = svd_scores(ratings_df, [m["id"] for m in movies])

    # ── 3. Hybrid blend ───────────────────────────────────────────
    hybrid = w_content * c_scores + w_collab * cf_scores

    # ── 4. Build result DataFrame ─────────────────────────────────
    records = []
    for i, m in enumerate(movies):
        genres = [GENRE_MAP.get(gid["id"], gid["name"]) for gid in m.get("genres", [])]
        credits = m.get("credits", {}) or {}
        director = next(
            (c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"),
            "—",
        )
        cast = [c["name"] for c in credits.get("cast", [])[:3]]
        trailer_key = None
        videos = m.get("videos", {}).get("results", [])
        for vid in videos:
            if vid.get("site") == "YouTube" and vid.get("type") == "Trailer":
                trailer_key = vid.get("key")
                break
        records.append({
            "id":           m["id"],
            "title":        m.get("title", "Unknown"),
            "year":         str(m.get("release_date", ""))[:4] or "—",
            "vote_average": round(m.get("vote_average", 0), 1),
            "vote_count":   m.get("vote_count", 0),
            "overview":     m.get("overview", ""),
            "poster_path":  m.get("poster_path"),
            "genres":       genres,
            "director":     director,
            "cast":         cast,
            "score_content": round(float(c_scores[i]), 4),
            "score_collab":  round(float(cf_scores[i]), 4),
            "score_hybrid":  round(float(hybrid[i]), 4),
            "trailer_key": trailer_key,
        })

    df = pd.DataFrame(records)
    df = df.sort_values("score_hybrid", ascending=False).head(top_n).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# CARD RENDERER
# ─────────────────────────────────────────────
def render_card(row: pd.Series, rank: int):
    hybrid  = row.score_hybrid
    content = row.score_content
    collab  = row.score_collab

    if hybrid >= 0.45:
        badge_cls, badge_lbl = "badge-hybrid",  "Hybrid"
    elif content > collab:
        badge_cls, badge_lbl = "badge-content", "Content"
    else:
        badge_cls, badge_lbl = "badge-collab",  "Collab"

    genre_chips = "".join(
        f'<span class="chip chip-genre">{g}</span>' for g in row.genres[:3]
    )
    cast_chips = "".join(
        f'<span class="chip chip-person">{c}</span>' for c in row.cast
    )
    director_chip = (
        f'<span class="chip chip-person">🎬 {row.director}</span>'
        if row.director != "—" else ""
    )

    def bar(pct: float, cls: str, label: str):
        w = int(pct * 100)
        v = f"{pct:.2f}"
        # Collapsed to a single line to prevent Streamlit from treating it as a Markdown code block
        return f'<div class="score-line"><span class="score-label">{label}</span><div class="score-bar-bg"><div class="score-bar-fill {cls}" style="width:{w}%"></div></div><span class="score-val">{v}</span></div>'

    st.markdown(f"""
    <div class="movie-card">
        <div class="card-poster-wrap">
            <img src="{poster_url(row.poster_path)}" alt="{row.title}" loading="lazy"/>
            <div class="card-rank-badge">#{rank}</div>
            <div class="card-type-badge {badge_cls}">{badge_lbl}</div>
        </div>
        <div class="card-body">
            <p class="card-title">{row.title}</p>
            <div class="card-meta">
                <span class="year">{row.year}</span>
                <span class="dot">·</span>
                <span class="vote">★ {row.vote_average}</span>
                <span class="dot">·</span>
                <span style="color:#6a6058">{row.vote_count:,} votes</span>
            </div>
            <div class="chip-row">{genre_chips}</div>
            <div class="chip-row">{director_chip}{cast_chips}</div>
            <p class="card-overview">{row.overview}</p>
            <div class="score-row">
                {bar(hybrid,  "fill-hybrid",  "Hybrid")}
                {bar(content, "fill-content", "Content")}
                {bar(collab,  "fill-collab",  "Collab")}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if row.trailer_key:
        with st.expander("▶️ Watch Trailer"):
            st.video(f"https://www.youtube.com/watch?v={row.trailer_key}")


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>CineMatch</h1>
    <p>Hybrid recommendations · Content-based × Collaborative filtering · Powered by TMDB</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    tmdb_key = st.secrets["TMDB_KEY"]

    st.markdown("---")
    st.markdown("### 🎯 Discovery Mode")

    mode = st.radio(
        "Recommendation strategy",
        ["Genre Explorer", "Similar to a Movie"],
        index=0,
    )

    query_title = ""
    if mode == "Similar to a Movie":
        query_title = st.text_input("Movie title to match", placeholder="e.g. Inception")

    selected_genres = st.multiselect(
        "Genre filter",
        options=GENRES_LIST,
        default=["Action", "Science Fiction"],
    )

    st.markdown("---")
    st.markdown("### ⚖️ Algorithm Weights")

    w_content = st.slider(
        "Content-based weight",
        min_value=0.0, max_value=1.0, value=0.55, step=0.05,
        help="Similarity via genres, director, cast & keywords",
    )
    w_collab = round(1.0 - w_content, 2)
    st.caption(f"Collaborative weight auto-set to **{w_collab}**")

    st.markdown("---")
    st.markdown("### 🔧 Settings")

    top_n      = st.slider("Results to show", 3, 18, 9, 3)
    min_rating = st.slider("Minimum TMDB rating", 0.0, 10.0, 5.0, 0.5)
    min_votes  = st.number_input("Minimum vote count", min_value=0, value=100, step=50)

    go = st.button("✨ Find My Movies")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if not tmdb_key:
    st.info("👈 Enter your **TMDB API key** in the sidebar to get started. "
            "Free keys available at [themoviedb.org](https://www.themoviedb.org/settings/api).")
    st.stop()

if go or ("results_df" in st.session_state):
    if go:
        if not selected_genres and mode == "Genre Explorer":
            st.warning("Please select at least one genre.")
            st.stop()

        with st.spinner("Fetching movies from TMDB…"):
            try:
                genre_ids = genre_ids_for_names(selected_genres)
                # Fetch two pages for a richer pool
                raw_movies: list[dict] = []
                for pg in [1, 2]:
                    api_response = tmdb_discover(tmdb_key, genre_ids, page=pg)
                    raw_movies += api_response.get("results", [])
    
                # Filter by rating / votes
                raw_movies = [
                    m for m in raw_movies
                    if m.get("vote_average", 0) >= min_rating
                    and m.get("vote_count", 0)   >= min_votes
                ]

                if not raw_movies:
                    st.warning("No movies matched your filters. Try relaxing the rating/vote thresholds.")
                    st.stop()

                # Shuffle for variety then take a pool for detail fetching
                random.shuffle(raw_movies)
                pool = raw_movies[:min(40, len(raw_movies))]

            except requests.HTTPError as e:
                if e.response.status_code == 401:
                    st.error("Invalid TMDB API key. Please check your key.")
                else:
                    st.error(f"TMDB error: {e}")
                st.stop()

        with st.spinner("Loading full metadata (credits, keywords)…"):
            detailed: list[dict] = []
            prog = st.progress(0)
            for i, m in enumerate(pool):
                d = tmdb_movie_details(tmdb_key, m["id"])
                time.sleep(0.05)
                if d:
                    detailed.append(d)
                prog.progress((i + 1) / len(pool))
            prog.empty()

            if not detailed:
                st.warning("Could not load detailed metadata.")
                st.stop()

        # Query movie for "Similar to" mode
        query_movie = None
        if mode == "Similar to a Movie" and query_title:
            with st.spinner(f"Searching for '{query_title}'…"):
                results = tmdb_search(tmdb_key, query_title)
                if results:
                    qm = tmdb_movie_details(tmdb_key, results[0]["id"])
                    if qm:
                        query_movie = qm

        with st.spinner("Running hybrid algorithm…"):
            df = hybrid_recommend(
                detailed,
                query_movie=query_movie,
                genre_names=selected_genres,
                w_content=w_content,
                w_collab=w_collab,
                top_n=top_n,
            )
            st.session_state["results_df"] = df
            st.session_state["query_movie"] = query_movie
    else:
        df = st.session_state["results_df"]
        query_movie = st.session_state.get("query_movie")

    # ── Stats bar ──────────────────────────────────────────────────
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Movies Found", len(df))
    with col_b:
        st.metric("Avg Hybrid Score", f"{df.score_hybrid.mean():.3f}")
    with col_c:
        st.metric("Avg TMDB Rating", f"{df.vote_average.mean():.1f} ★")
    with col_d:
        top_mode = "Content" if df.score_content.mean() > df.score_collab.mean() else "Collab"
        st.metric("Dominant Signal", top_mode)

    # ── Query reference ───────────────────────────────────────────
    if query_movie:
        credits = query_movie.get("credits", {}) or {}
        dir_name = next(
            (c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"),
            "—",
        )
        st.markdown(f"""
        <div style="background:rgba(255,154,60,.08);border:1px solid rgba(255,154,60,.2);
                    border-radius:12px;padding:1rem 1.4rem;margin-bottom:1.5rem;
                    display:flex;gap:1rem;align-items:center">
            <img src="{poster_url(query_movie.get('poster_path'))}"
                 style="width:52px;height:78px;object-fit:cover;border-radius:6px"/>
            <div>
                <div style="font-family:'Playfair Display',serif;font-size:1.05rem;
                            color:#ffd77a;font-weight:700">{query_movie.get('title','')}</div>
                <div style="font-size:.8rem;color:#8a8070;margin-top:2px">
                    Matching similar movies · Director: {dir_name}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Results grid ──────────────────────────────────────────────
    label = (
        f"Top {len(df)} recommendations · Similar to <em>{query_movie['title']}</em>"
        if query_movie else
        f"Top {len(df)} recommendations · {', '.join(selected_genres) or 'All genres'}"
    )
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)

    COLS = 3
    for row_start in range(0, len(df), COLS):
        cols = st.columns(COLS, gap="medium")
        for ci, col in enumerate(cols):
            idx = row_start + ci
            if idx < len(df):
                with col:
                    render_card(df.iloc[idx], rank=idx + 1)

    # ── Score breakdown chart (expander) ──────────────────────────
    with st.expander("📊 Score breakdown table"):
        display_cols = ["title", "year", "vote_average", "score_hybrid", "score_content", "score_collab", "director"]
        st.dataframe(
            df[display_cols].rename(columns={
                "vote_average":  "TMDB ★",
                "score_hybrid":  "Hybrid",
                "score_content": "Content",
                "score_collab":  "Collab",
            }),
            use_container_width=True,
            hide_index=True,
        )

else:
    # Welcome state
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;color:#4a4038">
        <div style="font-size:4rem;margin-bottom:1rem">🎬</div>
        <div style="font-family:'Playfair Display',serif;font-size:1.6rem;
                    color:#6a5a48;margin-bottom:.6rem">Ready when you are</div>
        <div style="font-size:.9rem;max-width:440px;margin:0 auto;line-height:1.7">
            Configure your preferences in the sidebar, then hit
            <strong style="color:#ff9a3c">Find My Movies</strong> to let the hybrid
            engine work its magic.
        </div>
        <div style="margin-top:2.5rem;display:flex;justify-content:center;gap:2rem;
                    font-size:.8rem;color:#3a3028">
            <div>🔵 Content-Based<br><span style="color:#4a4038">Genres · Director · Cast</span></div>
            <div style="color:#2a2018">×</div>
            <div>🟣 Collaborative<br><span style="color:#4a4038">SVD Matrix Factorisation</span></div>
            <div style="color:#2a2018">=</div>
            <div>🟠 Hybrid Score<br><span style="color:#4a4038">Blended ranking</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
