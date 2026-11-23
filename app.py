"""
app.py
======
CineMatch – Intelligent Movie Recommendation System
Main Streamlit application entry point.

Run with:
    streamlit run app.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Project root on sys.path (so `src` package imports work)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import (
    all_files_present,
    dataset_status,
    download_instructions,
    load_links,
    load_movies,
    load_ratings,
    load_tags,
)
from src.movie_details import get_movie_details, search_movies
from src.preprocessing import build_movie_profiles
from src.recommender import CollaborativeRecommender, MovieRecommender
from src.user_preferences import (
    add_to_favourites,
    add_to_watchlist,
    is_favourite,
    is_in_watchlist,
    load_preferences,
    remove_from_favourites,
    remove_from_watchlist,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS – dark, Netflix-inspired theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Global ---- */
    .stApp { background-color: #141414; color: #FFFFFF; }

    /* ---- Hero banner ---- */
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid #E50914;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        color: #E50914;
        letter-spacing: -1px;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #AAAAAA;
        margin-top: 0.4rem;
    }

    /* ---- Movie card ---- */
    .movie-card {
        background: #1F1F1F;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #E50914;
        transition: transform 0.15s ease;
    }
    .movie-card:hover { border-left-color: #FF6B35; }
    .card-title { font-size: 1.1rem; font-weight: 700; color: #FFFFFF; margin: 0; }
    .card-meta  { font-size: 0.85rem; color: #AAAAAA; margin-top: 0.2rem; }
    .card-score {
        display: inline-block;
        background: #E50914;
        color: white;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-top: 0.4rem;
    }
    .genre-tag {
        display: inline-block;
        background: #2A2A2A;
        color: #E50914;
        border: 1px solid #E50914;
        border-radius: 20px;
        padding: 2px 8px;
        font-size: 0.75rem;
        margin: 2px;
    }
    .feature-pill {
        display: inline-block;
        background: #0f3460;
        color: #90CAF9;
        border-radius: 20px;
        padding: 2px 8px;
        font-size: 0.72rem;
        margin: 2px;
    }

    /* ---- Section header ---- */
    .section-header {
        font-size: 1.4rem;
        font-weight: 800;
        color: #FFFFFF;
        border-bottom: 2px solid #E50914;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }

    /* ---- Stats cards ---- */
    .stat-card {
        background: #1F1F1F;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #333;
    }
    .stat-number { font-size: 2rem; font-weight: 900; color: #E50914; }
    .stat-label  { font-size: 0.85rem; color: #AAAAAA; }

    /* ---- Rating stars ---- */
    .star-rating { color: #FFD700; font-size: 1rem; }

    /* ---- Info box ---- */
    .info-box {
        background: #1a2a3a;
        border-left: 3px solid #4FC3F7;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #B0BEC5;
    }

    /* ---- Sidebar ---- */
    .css-1d391kg, [data-testid="stSidebar"] { background-color: #1F1F1F !important; }

    /* ---- Streamlit overrides ---- */
    .stSelectbox label, .stTextInput label, .stMultiSelect label { color: #CCCCCC !important; }
    div[data-baseweb="select"] { background: #2A2A2A !important; }
    .stButton > button {
        background: #E50914;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 700;
    }
    .stButton > button:hover { background: #c40811; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loading (cached so it runs only once per session)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading dataset …")
def load_all_data():
    """Load, clean, and preprocess the full MovieLens dataset."""
    movies_raw = load_movies()
    ratings_df = load_ratings()
    tags_df = load_tags()
    links_df = load_links()
    movies_df = build_movie_profiles(movies_raw, tags_df)
    return movies_df, ratings_df, tags_df, links_df


@st.cache_resource(show_spinner="Building recommendation engine …")
def build_recommender(movies_df: pd.DataFrame) -> MovieRecommender:
    """Build and cache the TF-IDF recommendation model."""
    return MovieRecommender(movies_df)


@st.cache_resource(show_spinner="Building collaborative filtering model …")
def build_collaborative(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> CollaborativeRecommender:
    return CollaborativeRecommender(ratings_df, movies_df)


# ---------------------------------------------------------------------------
# Helper UI functions
# ---------------------------------------------------------------------------

def render_movie_card(movie: dict, rank: int = None) -> None:
    """Render a single movie recommendation card."""
    genres_html = "".join(f'<span class="genre-tag">{g}</span>' for g in movie.get("genres_list", []))
    year_str = f" ({movie['year']})" if movie.get("year") else ""
    score_pct = int(movie["score"] * 100)

    features = movie.get("overlap_features", "")
    if features and features != "No shared features":
        pills = "".join(
            f'<span class="feature-pill">{f.strip()}</span>'
            for f in features.split(",")[:6]
        )
        features_html = f"<div style='margin-top:0.4rem'><span style='color:#888;font-size:0.75rem'>Key features: </span>{pills}</div>"
    else:
        features_html = ""

    rank_badge = f'<span style="color:#888;font-size:0.85rem">#{rank} &nbsp;</span>' if rank else ""

    st.markdown(
        f"""
        <div class="movie-card">
            <p class="card-title">{rank_badge}{movie['title']}{year_str}</p>
            <div class="card-meta">{genres_html}</div>
            {features_html}
            <span class="card-score">Similarity: {score_pct}%</span>
            <span style="font-size:0.75rem;color:#888;margin-left:0.5rem">
                (algorithmic content match — not a rating prediction)
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_detail_card(details: dict, ratings_df: pd.DataFrame) -> None:
    """Render detailed movie information."""
    year_str = f"({details['year']})" if details.get("year") else ""
    genres_html = "".join(f'<span class="genre-tag">{g}</span>' for g in details.get("genres_list", []))

    stars = ""
    if details.get("avg_rating") is not None:
        full = int(details["avg_rating"] / 2)
        stars = "⭐" * full + "☆" * (5 - full)

    tags_html = ""
    if details.get("tags"):
        tags_html = "".join(
            f'<span class="feature-pill">{t}</span>'
            for t in details["tags"][:20]
        )

    imdb_link = ""
    if details.get("imdb_id") and str(details["imdb_id"]) != "nan":
        imdb_link = f'<a href="https://www.imdb.com/title/tt{details["imdb_id"].zfill(7)}/" target="_blank" style="color:#F5C518">🔗 IMDb</a>'

    st.markdown(
        f"""
        <div style="background:#1a1a2e;border-radius:16px;padding:1.5rem;border:1px solid #E50914;margin-bottom:1rem">
            <h2 style="color:#FFFFFF;margin:0">{details['clean_title']} <span style="color:#888;font-size:1rem;font-weight:400">{year_str}</span></h2>
            <div style="margin:0.5rem 0">{genres_html}</div>
            <div style="margin:0.3rem 0;font-size:0.9rem;color:#AAAAAA">Movie ID: {details['movieId']} &nbsp;|&nbsp; {imdb_link}</div>
            <hr style="border-color:#333;margin:0.8rem 0">
            <div style="display:flex;gap:2rem;flex-wrap:wrap">
                <div>
                    <span style="color:#888;font-size:0.8rem">Avg. Rating</span><br>
                    <span class="star-rating">{stars}</span>
                    <span style="color:#FFD700;font-weight:700"> {details['avg_rating'] or 'N/A'}</span>
                    <span style="color:#888;font-size:0.8rem">/5.0</span>
                </div>
                <div>
                    <span style="color:#888;font-size:0.8rem">Total Ratings</span><br>
                    <span style="color:#FFFFFF;font-weight:700;font-size:1.1rem">{details['num_ratings']:,}</span>
                </div>
            </div>
            {"<div style='margin-top:0.8rem'><span style='color:#888;font-size:0.8rem'>User Tags</span><br>" + tags_html + "</div>" if tags_html else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------

def page_home(movies_df, ratings_df, tags_df, links_df, recommender):
    """Home page with branding, stats, and quick search."""
    # Hero banner
    st.markdown(
        """
        <div class="hero-banner">
            <p class="hero-title">🎬 CineMatch</p>
            <p class="hero-subtitle">Intelligent Movie Recommendations powered by TF-IDF & Cosine Similarity</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Stats row
    total_movies = len(movies_df)
    total_ratings = len(ratings_df)
    tagged_movies = movies_df[movies_df["tags_text"].str.strip() != ""].shape[0]
    vocab_size = recommender.get_vocab_size()

    c1, c2, c3, c4 = st.columns(4)
    for col, num, label in [
        (c1, f"{total_movies:,}", "Movies"),
        (c2, f"{total_ratings:,}", "Ratings"),
        (c3, f"{tagged_movies:,}", "Tagged Movies"),
        (c4, f"{vocab_size:,}", "TF-IDF Features"),
    ]:
        col.markdown(
            f'<div class="stat-card"><div class="stat-number">{num}</div>'
            f'<div class="stat-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick search on home page
    st.markdown('<p class="section-header">🔍 Quick Search</p>', unsafe_allow_html=True)
    query = st.text_input("Search for a movie …", placeholder="e.g. Interstellar, Toy Story, Batman …")
    if query:
        results = search_movies(query, movies_df, max_results=10)
        if results.empty:
            st.warning(f'No movies found matching "{query}". Try a different search term.')
        else:
            st.success(f"Found {len(results)} result(s):")
            for _, row in results.iterrows():
                year = f" ({int(row['year'])})" if pd.notna(row.get("year")) else ""
                genres = " | ".join(row["genres_list"][:3]) if row["genres_list"] else "Unknown"
                st.markdown(
                    f'<div class="movie-card">'
                    f'<p class="card-title">{row["title"]}{year}</p>'
                    f'<p class="card-meta">{genres}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Algorithm info box
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-box">
            <strong>How it works:</strong> CineMatch builds a textual profile for each movie by combining its
            genres and user tags. Each profile is converted into a numerical vector using <strong>TF-IDF</strong>
            (Term Frequency–Inverse Document Frequency). Movies are compared using <strong>Cosine Similarity</strong>
            — the smaller the angle between two vectors, the more similar the movies.
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_discover(movies_df, ratings_df, tags_df, links_df, recommender):
    """Discover page: search, select a movie, view recommendations."""
    st.markdown('<p class="section-header">🎯 Discover Movies</p>', unsafe_allow_html=True)

    # Algorithm selector
    algo = st.radio(
        "Recommendation algorithm:",
        ["TF-IDF + Cosine Similarity", "Jaccard Similarity (Genre-based)"],
        horizontal=True,
        help="TF-IDF uses genres AND tags; Jaccard uses genres only."
    )

    top_n = st.slider("Number of recommendations", 5, 20, 10)

    st.markdown("---")

    # Search
    query = st.text_input("🔍 Search for a movie:", placeholder="Type a title …")

    if not query:
        st.info("Start typing a movie title above to search.")
        return

    results = search_movies(query, movies_df, max_results=20)

    if results.empty:
        st.warning(f'No movies found for "{query}". Try a shorter or different term.')
        return

    # Build selection options
    options = {}
    for _, row in results.iterrows():
        year = f" ({int(row['year'])})" if pd.notna(row.get("year")) else ""
        label = f"{row['title']}{year}"
        options[label] = int(row["movieId"])

    selected_label = st.selectbox("Select a movie:", list(options.keys()))
    selected_id = options[selected_label]

    # Movie details
    details = get_movie_details(selected_id, movies_df, ratings_df, tags_df, links_df)
    if details:
        st.markdown("#### Selected Movie Details")
        render_detail_card(details, ratings_df)

        # Watchlist / Favourites buttons
        bc1, bc2 = st.columns(2)
        with bc1:
            if is_favourite(selected_id):
                if st.button("💔 Remove from Favourites"):
                    remove_from_favourites(selected_id)
                    st.rerun()
            else:
                if st.button("❤️ Add to Favourites"):
                    add_to_favourites(selected_id)
                    st.success("Added to favourites!")
        with bc2:
            if is_in_watchlist(selected_id):
                if st.button("🗑️ Remove from Watchlist"):
                    remove_from_watchlist(selected_id)
                    st.rerun()
            else:
                if st.button("📌 Add to Watchlist"):
                    add_to_watchlist(selected_id)
                    st.success("Added to watchlist!")

    # Generate recommendations
    st.markdown("---")
    st.markdown('<p class="section-header">🤖 Recommendations</p>', unsafe_allow_html=True)

    with st.spinner("Calculating similarities …"):
        if algo == "TF-IDF + Cosine Similarity":
            recs = recommender.recommend_by_movie(selected_id, top_n=top_n)
            algo_note = "Based on TF-IDF content vectors and cosine similarity"
        else:
            recs = recommender.recommend_by_jaccard(selected_id, top_n=top_n)
            algo_note = "Based on Jaccard similarity of genre sets"

    if not recs:
        st.warning("Could not generate recommendations for this movie.")
        return

    st.caption(f"Algorithm: {algo_note}")

    for rank, movie in enumerate(recs, start=1):
        c_card, c_actions = st.columns([4, 1])
        with c_card:
            render_movie_card(movie, rank=rank)
        with c_actions:
            st.markdown("<br>", unsafe_allow_html=True)
            mid = movie["movieId"]
            if st.button("📌 Watchlist", key=f"wl_{mid}_{rank}"):
                add_to_watchlist(mid)
                st.success("Added!")
            if st.button("❤️ Favourite", key=f"fav_{mid}_{rank}"):
                add_to_favourites(mid)
                st.success("Added!")


def page_recommendations(movies_df, ratings_df, tags_df, links_df, recommender, collab_recommender):
    """Multiple favourites recommendation page."""
    st.markdown('<p class="section-header">🌟 My Taste – Multi-Favourite Recommendations</p>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
            Select <strong>2–10 movies</strong> you enjoy. CineMatch will average their TF-IDF profiles
            to build a combined taste vector and find movies that match all your preferences together.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Build a full title list for the multiselect
    title_map = {}
    for _, row in movies_df.iterrows():
        year = f" ({int(row['year'])})" if pd.notna(row.get("year")) else ""
        label = f"{row['title']}{year} [ID:{row['movieId']}]"
        title_map[label] = int(row["movieId"])

    selected_labels = st.multiselect(
        "Choose your favourite movies:",
        list(title_map.keys()),
        max_selections=10,
        placeholder="Search and select movies …",
    )

    if len(selected_labels) < 2:
        st.info("Select at least 2 movies to generate combined recommendations.")
        return

    selected_ids = [title_map[lbl] for lbl in selected_labels]

    algo = st.radio(
        "Algorithm:",
        ["TF-IDF + Cosine (Combined Profile)", "Collaborative Filtering"],
        horizontal=True,
    )
    top_n = st.slider("Recommendations to show", 5, 20, 10, key="multi_topn")

    if st.button("🚀 Generate Recommendations"):
        with st.spinner("Building combined taste profile …"):
            if algo == "TF-IDF + Cosine (Combined Profile)":
                recs = recommender.recommend_by_favourites(selected_ids, top_n=top_n)
                note = "Combined TF-IDF average profile → cosine similarity"
            else:
                # For collaborative, use the first selected movie as proxy
                recs = collab_recommender.recommend(selected_ids[0], top_n=top_n)
                note = "Collaborative filtering (item-item cosine on ratings matrix)"

        if not recs:
            st.warning("Could not generate recommendations. Try different movies.")
            return

        st.success(f"Top {len(recs)} recommendations based on your combined taste:")
        st.caption(f"Algorithm: {note}")
        for rank, movie in enumerate(recs, start=1):
            render_movie_card(movie, rank=rank)


def page_genre_explorer(movies_df, ratings_df, tags_df, links_df):
    """Genre exploration page."""
    st.markdown('<p class="section-header">🎭 Genre Explorer</p>', unsafe_allow_html=True)

    # Collect all unique genres
    all_genres = sorted({
        g for genres_list in movies_df["genres_list"] for g in genres_list
    })

    selected_genres = st.multiselect(
        "Select genres to explore:",
        all_genres,
        default=["Action"],
        placeholder="Pick one or more genres …",
    )

    sort_by = st.selectbox("Sort by:", ["Title (A-Z)", "Year (Newest)", "Year (Oldest)"])

    if not selected_genres:
        st.info("Select at least one genre above.")
        return

    # Filter movies matching ALL selected genres (intersection)
    match_all = st.checkbox("Match ALL selected genres (strict)", value=False)

    if match_all:
        mask = movies_df["genres_list"].apply(
            lambda gl: all(g in gl for g in selected_genres)
        )
    else:
        mask = movies_df["genres_list"].apply(
            lambda gl: any(g in gl for g in selected_genres)
        )

    filtered = movies_df[mask].copy()

    # Sort
    if sort_by == "Title (A-Z)":
        filtered = filtered.sort_values("title")
    elif sort_by == "Year (Newest)":
        filtered = filtered.sort_values("year", ascending=False, na_position="last")
    else:
        filtered = filtered.sort_values("year", ascending=True, na_position="last")

    st.markdown(f"**{len(filtered)} movies** found in selected genre(s).")
    st.markdown("---")

    for _, row in filtered.head(50).iterrows():
        year = f" ({int(row['year'])})" if pd.notna(row.get("year")) else ""
        genres_html = "".join(f'<span class="genre-tag">{g}</span>' for g in row["genres_list"])
        st.markdown(
            f'<div class="movie-card">'
            f'<p class="card-title">{row["title"]}{year}</p>'
            f'<div class="card-meta">{genres_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if len(filtered) > 50:
        st.caption(f"Showing first 50 of {len(filtered)} results.")


def page_watchlist(movies_df, ratings_df, tags_df, links_df, recommender):
    """Watchlist and favourites management page."""
    st.markdown('<p class="section-header">📋 My Watchlist & Favourites</p>', unsafe_allow_html=True)

    prefs = load_preferences()
    tab_wl, tab_fav = st.tabs(["📌 Watchlist", "❤️ Favourites"])

    def render_saved_list(movie_ids: list, list_key: str, remove_fn):
        if not movie_ids:
            st.info("Nothing saved here yet. Add movies from the Discover page.")
            return
        for mid in movie_ids:
            details = get_movie_details(mid, movies_df, ratings_df, tags_df, links_df)
            if not details:
                continue
            year = f" ({details['year']})" if details.get("year") else ""
            genres_html = "".join(f'<span class="genre-tag">{g}</span>' for g in details.get("genres_list", []))
            rating_str = (
                f"⭐ {details['avg_rating']}/5 ({details['num_ratings']:,} ratings)"
                if details.get("avg_rating") else "No ratings"
            )
            c_info, c_btn = st.columns([4, 1])
            with c_info:
                st.markdown(
                    f'<div class="movie-card">'
                    f'<p class="card-title">{details["title"]}{year}</p>'
                    f'<div class="card-meta">{genres_html}</div>'
                    f'<div class="card-meta">{rating_str}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with c_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Remove", key=f"remove_{list_key}_{mid}"):
                    remove_fn(mid)
                    st.rerun()

    with tab_wl:
        st.markdown(f"**{len(prefs['watchlist'])} movie(s)** in your watchlist.")
        render_saved_list(prefs["watchlist"], "wl", remove_from_watchlist)

    with tab_fav:
        st.markdown(f"**{len(prefs['favourites'])} movie(s)** in your favourites.")
        render_saved_list(prefs["favourites"], "fav", remove_from_favourites)

        if len(prefs["favourites"]) >= 2:
            st.markdown("---")
            st.markdown("#### 🤖 Recommendations based on your favourites")
            if st.button("Generate from my favourites"):
                recs = recommender.recommend_by_favourites(prefs["favourites"], top_n=10)
                for rank, movie in enumerate(recs, start=1):
                    render_movie_card(movie, rank=rank)


def page_about():
    """About page explaining the project and algorithms."""
    st.markdown('<p class="section-header">ℹ️ About CineMatch</p>', unsafe_allow_html=True)

    st.markdown(
        """
        ## CineMatch – Intelligent Movie Recommendation System

        CineMatch is a Python mini-project that demonstrates **content-based filtering**
        using real movie data from the [MovieLens dataset](https://grouplens.org/datasets/movielens/).

        ---

        ### 📐 Recommendation Algorithms

        #### 1. TF-IDF (Term Frequency–Inverse Document Frequency)
        Each movie gets a *profile* by combining its genres and user tags.
        TF-IDF converts these profiles into numerical vectors:

        | Term | Meaning |
        |------|---------|
        | **TF** | How often a word appears in *this* movie's profile |
        | **IDF** | How rare that word is across *all* movies |
        | **TF × IDF** | High weight for distinctive, movie-specific words |

        > Words like *"Drama"* appear in thousands of movies → low IDF → low weight.
        > Words like *"heist"* appear rarely → high IDF → high weight.

        #### 2. Cosine Similarity
        After TF-IDF vectorisation, each movie is a point in high-dimensional space.
        Cosine similarity measures the **angle** between two movie vectors:

        ```
        similarity = (A · B) / (||A|| × ||B||)
        ```

        Score = 1.0 → identical profiles. Score = 0.0 → no shared features.

        #### 3. Jaccard Similarity (Genre-based)
        A simpler baseline using **genre sets**:

        ```
        J(A, B) = |A ∩ B| / |A ∪ B|
        ```

        Example:
        - Movie A genres: {Action, Thriller}
        - Movie B genres: {Action, Drama}
        - Intersection: {Action} → size 1
        - Union: {Action, Thriller, Drama} → size 3
        - Jaccard score: 1/3 ≈ 0.33

        ---

        ### 🏗️ Project Architecture

        ```
        Movie-Recommendation-System/
        ├── app.py               ← Streamlit interface
        ├── download_data.py     ← Dataset download helper
        ├── src/
        │   ├── data_loader.py   ← CSV loading & validation
        │   ├── preprocessing.py ← Data cleaning & profile building
        │   ├── similarity.py    ← Jaccard & cosine formulas
        │   ├── recommender.py   ← TF-IDF model & recommendation logic
        │   ├── movie_details.py ← Movie metadata & search
        │   └── user_preferences.py ← Watchlist / favourites (JSON)
        └── tests/               ← Pytest test suite
        ```

        ---

        ### 📦 Dataset
        [MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/)
        by GroupLens Research, University of Minnesota.

        Contains ~9,700 movies and ~100,000 ratings from ~600 users.

        ---

        ### ⚠️ Limitations
        - Cold-start: new movies with no tags rely on genres only.
        - Genre labels are coarse; two "Action" movies can be very different.
        - Similarity scores are content matches, **not** rating predictions.
        - No real user authentication; preferences are stored locally.
        """
    )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Sidebar navigation ----
    with st.sidebar:
        st.markdown("## 🎬 CineMatch")
        st.markdown("---")
        page = st.radio(
            "Navigation",
            ["🏠 Home", "🎯 Discover", "🌟 For You", "🎭 Genres", "📋 My List", "ℹ️ About"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("Built with Streamlit · MovieLens dataset · TF-IDF + Cosine Similarity")

    # ---- Dataset check ----
    if not all_files_present():
        with st.spinner("📦 Downloading MovieLens dataset for first launch (~10 seconds)..."):
            try:
                from download_data import download_and_extract
                download_and_extract()
                st.rerun()
            except Exception as exc:
                status = dataset_status()
                st.error(f"⚠️ Dataset Missing (Auto-download failed: {exc})")
                st.markdown(download_instructions())
                st.markdown("**File status:**")
                for name, present in status.items():
                    icon = "✅" if present else "❌"
                    st.markdown(f"{icon} `{name}.csv`")
                st.stop()

    # ---- Load data ----
    try:
        movies_df, ratings_df, tags_df, links_df = load_all_data()
    except Exception as exc:
        st.error(f"Error loading dataset: {exc}")
        st.stop()

    # ---- Build models ----
    try:
        recommender = build_recommender(movies_df)
    except Exception as exc:
        st.error(f"Error building recommendation model: {exc}")
        st.stop()

    try:
        collab_recommender = build_collaborative(ratings_df, movies_df)
    except Exception as exc:
        st.warning(f"Collaborative filtering unavailable: {exc}")
        collab_recommender = None

    # ---- Route to page ----
    if page == "🏠 Home":
        page_home(movies_df, ratings_df, tags_df, links_df, recommender)
    elif page == "🎯 Discover":
        page_discover(movies_df, ratings_df, tags_df, links_df, recommender)
    elif page == "🌟 For You":
        page_recommendations(movies_df, ratings_df, tags_df, links_df, recommender, collab_recommender)
    elif page == "🎭 Genres":
        page_genre_explorer(movies_df, ratings_df, tags_df, links_df)
    elif page == "📋 My List":
        page_watchlist(movies_df, ratings_df, tags_df, links_df, recommender)
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
