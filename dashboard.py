"""Streamlit dashboard for visualising synthesised disaster data.

Run this app with:

    streamlit run disaster_response/dashboard.py

The app allows uploading a CSV dataset, triggering the classification pipeline,
and displays progress, interactive charts, a map (heatmap + points), and rumors in a top-to-down layout.
Geocoding is performed on the fly using geopy with cached results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import tempfile

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Import all necessary components to ensure complete pipeline
from disaster_response.config import load_config, CONFIG
from disaster_response.data_ingestion import load_dataset, Tweet
from disaster_response.headline_generation import generate_headline
from disaster_response.vector_db import VectorDB
from disaster_response.search_enrichment import get_contexts
from disaster_response.classification import classify_tweet

# Set up logging
logger = logging.getLogger(__name__)

# Load configuration
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)
SYN_JSON = ARTIFACTS_DIR / "synthesized_data.json"
RUMORS_JSON = ARTIFACTS_DIR / "rumors.json"
GEOCACHE_PATH = ARTIFACTS_DIR / "geocache.json"
FAISS_INDEX_DIR = ARTIFACTS_DIR / "faiss_index"

@st.cache_data(show_spinner=False)
def geocode_locations(records: List[Dict[str, Any]], country: str = "Bangladesh") -> List[Dict[str, Any]]:
    """Geocode location names to latitude and longitude with robust error handling."""
    geolocator = Nominatim(user_agent="disaster-response-dashboard")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=3, error_wait_seconds=2)
    enriched = []
    for rec in records:
        loc = rec.get("location")
        if not loc or not isinstance(loc, str):
            continue
        # Avoid duplicate country in query
        query = f"{loc.strip()}, {country.strip()}" if loc.strip().lower() != country.strip().lower() else country
        try:
            location = geocode(query, timeout=5)
            if location:
                rec = rec.copy()
                rec["latitude"] = location.latitude
                rec["longitude"] = location.longitude
                enriched.append(rec)
        except Exception as e:
            st.warning(f"Geocoding failed for {query}: {str(e)}")
            continue
    return enriched


def handle_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to a temporary path and return the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def process_tweets_dashboard(
    tweets: List[Tweet],
    vector_db: VectorDB,
    progress_callback: Optional[Callable[[float], None]] = None
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process each tweet through the complete pipeline following main.py exactly.
    
    This function replicates the exact same pipeline as in main.py:
    1. Generate headline for each tweet
    2. Get contexts using the headline
    3. Classify tweet with contexts
    4. Separate rumors from non-rumors
    
    Parameters
    ----------
    tweets: List[Tweet]
        List of tweets to process.
    vector_db: VectorDB
        The vector database for contextual search and storage.
    progress_callback: Optional[Callable[[float], None]]
        Optional callback to update progress (e.g., for UI).
    """
    rumors: List[Dict[str, Any]] = []
    synthesized: List[Dict[str, Any]] = []
    total = len(tweets)
    
    # Create a status placeholder for detailed progress
    status_placeholder = st.empty()
    
    for i, tweet in enumerate(tweets):
        try:
            # Update detailed status
            status_placeholder.text(f"Processing tweet {i+1}/{total}: Generating headline...")
            
            # Step 1: Generate a search-friendly headline (EXACTLY as in main.py)
            headline = generate_headline(tweet.tweet, tweet.timestamp)
            logger.info("Generated headline: %s", headline)
            
            # Update status
            status_placeholder.text(f"Processing tweet {i+1}/{total}: Retrieving contexts...")
            
            # Retrieve contexts from the vector database or via web search (OPTIMIZED)
            contexts = get_contexts(
                headline, 
                vector_db, 
                top_k=3, 
                search_if_insufficient=True,
                min_required_contexts=getattr(st.session_state, 'min_contexts', 2),
                relevance_threshold=getattr(st.session_state, 'relevance_threshold', 0.4)
            )
            
            # Update status
            status_placeholder.text(f"Processing tweet {i+1}/{total}: Classifying...")
            
            # Step 3: Classify the tweet and extract structured info (EXACTLY as in main.py)
            classification = classify_tweet(tweet, contexts)
            label = classification.get("label")
            
            if label == "rumor":
                # Only store the needed fields for rumours
                rumors.append({
                    "tweet": tweet.tweet,
                    "label": "rumor",
                    "reasoning": classification.get("reasoning"),
                    "confidence": classification.get("confidence"),
                })
            elif label == "non-rumor":
                # Remove label for synthesized data; include all structured fields
                structured = {k: v for k, v in classification.items() if k != "label"}
                structured["tweet"] = tweet.tweet
                synthesized.append(structured)
            else:
                logger.warning("Unknown classification label %s for tweet: %s", label, tweet.tweet)
                
        except Exception as e:
            logger.error(f"Error processing tweet {i+1}: {str(e)}")
            st.error(f"Error processing tweet {i+1}: {str(e)}")
            continue
        
        # Update progress callback
        if progress_callback:
            progress_callback((i + 1) / total)
    
    # Clear the detailed status
    status_placeholder.empty()
    
    return rumors, synthesized


def process_and_display_pipeline(file_path: str):
    """Handle the processing pipeline and display progress on the dashboard."""
    # Load tweets
    with st.spinner("Loading dataset..."):
        tweets = load_dataset(file_path)
        st.success(f"Loaded {len(tweets)} tweets from dataset")
    
    # Initialize vector database
    with st.spinner("Initializing vector database..."):
        vector_db = VectorDB()
        st.success("Vector database initialized")

    # Create progress tracking
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def progress_callback(progress: float):
        progress_bar.progress(progress)
        status_text.text(f"Processing tweets: {int(progress * 100)}% complete")

    # Show pipeline steps
    st.subheader("Pipeline Steps")
    steps_info = st.empty()
    steps_info.markdown("""
    **Complete Pipeline (following main.py exactly):**
    1. 🎯 **Generate headline** for each tweet to create search-friendly queries
    2. 🔍 **Retrieve contexts** using the headline from vector DB or web search  
    3. 🏷️ **Classify tweet** as rumor/non-rumor using contexts
    4. 📊 **Structure data** and separate rumors from disaster information
    """)

    # Run the complete pipeline
    with st.spinner("Running complete classification pipeline..."):
        try:
            rumors, synthesized = process_tweets_dashboard(tweets, vector_db, progress_callback)
            
            # Save to files for persistence (same as main.py)
            with open("rumors.json", "w", encoding="utf-8") as f:
                json.dump(rumors, f, indent=2, ensure_ascii=False)
            with open("synthesized_data.json", "w", encoding="utf-8") as f:
                json.dump(synthesized, f, indent=2, ensure_ascii=False)

            st.success("Pipeline complete! Data processed and saved.")
            
            # Show summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Tweets Processed", len(tweets))
            with col2:
                st.metric("Rumors Identified", len(rumors))
            with col3:
                st.metric("Disaster Information", len(synthesized))
                
        except Exception as e:
            st.error(f"Pipeline failed: {str(e)}")
            logger.error(f"Pipeline failed: {str(e)}")
            return [], []

    return rumors, synthesized


def main():
    st.set_page_config(page_title="Disaster Response Dashboard", layout="wide", initial_sidebar_state="expanded")
    
    # Load configuration
    try:
        global CONFIG
        CONFIG = load_config(".env")
        logging.basicConfig(level=logging.INFO)
    except Exception as e:
        st.error(f"Configuration error: {str(e)}")
    
    # Sidebar for configurations
    with st.sidebar:
        st.header("Settings")
        country = st.selectbox("Select Country for Geocoding", ["Bangladesh", "India", "Pakistan", "Nepal", "Other"], index=0)
        if country == "Other":
            country = st.text_input("Enter Country Name")
        
        st.header("🧠 Smart Search Settings")
        
        # Advanced settings
        with st.expander("Advanced Context Settings"):
            min_contexts = st.slider("Minimum Required Contexts", 1, 5, 2)
            relevance_threshold = st.slider("Relevance Threshold", 0.1, 0.9, 0.4, 0.1)
            st.caption("Higher threshold = more likely to search for fresh content")
        

        st.header("🚀 Pipeline Features")
        st.markdown("""
        **Optimized Pipeline:**
        - ✅ **Smart Search Decisions**: Only searches when needed
        - ✅ **Relevance Checking**: Evaluates context quality
        - ✅ **Topic Tracking**: Avoids redundant searches
        - ✅ **Search Caching**: Prevents duplicate API calls
        - ✅ **Headline Generation**: Creates better search queries
        """)

    st.title("Disaster Response Dashboard")
    st.markdown("""
    This dashboard allows to upload a disaster dataset (CSV), run the **complete classification pipeline** 
    , and visualize the results including needs distribution, severity levels, 
    geographic map (heatmap + points), and identified rumors.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Upload a Disaster Dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        tmp_path = handle_uploaded_file(uploaded_file)
        
        # Load preview
        try:
            if uploaded_file.name.endswith('.tsv'):
                df_preview = pd.read_csv(tmp_path, sep='\t')
            else:
                df_preview = pd.read_csv(tmp_path)
            
            st.subheader("Dataset Preview")
            st.dataframe(df_preview.head(10), use_container_width=True)
            
            # Show expected columns
            expected_cols = ["user_id", "tweet", "location", "timestamp", "followers"]
            actual_cols = list(df_preview.columns)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Expected columns:**", expected_cols)
            with col2:
                st.write("**Your columns:**", actual_cols)
                
            missing_cols = set(expected_cols) - set(actual_cols)
            if missing_cols:
                st.warning(f"Missing columns: {missing_cols}")
            else:
                st.success("✅ All required columns present!")

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return

        # Store settings in session state
        st.session_state.min_contexts = locals().get('min_contexts', 2)
        st.session_state.relevance_threshold = locals().get('relevance_threshold', 0.4)

        # Button to trigger pipeline
        if st.button("🚀 Run Optimized Classification Pipeline", type="primary"):
            rumors, synthesized = process_and_display_pipeline(tmp_path)
            st.session_state.rumors = rumors
            st.session_state.synthesized = synthesized
            
           
            
            st.rerun()

    # Load data from session state
    synthesized = st.session_state.get("synthesized", [])
    rumors = st.session_state.get("rumors", [])

    if synthesized or rumors:
        st.markdown("---")
        st.header("📊 Results Visualization")

    if synthesized:
        # Prepare DataFrame and explode needs
        df = pd.DataFrame(synthesized)
        
        # Handle need_type field more robustly
        if 'need_type' in df.columns:
            df['need_type'] = df['need_type'].apply(lambda x: x if isinstance(x, list) else (x.split(',') if isinstance(x, str) else []))
            df_exploded = df.explode('need_type')
            df_exploded['need_type'] = df_exploded['need_type'].str.lower().str.strip()
        else:
            df_exploded = df
            st.warning("No 'need_type' column found in synthesized data")
        
        # Map severity to numeric for heatmap and sizing
        severity_map = {'low': 1, 'medium': 2, 'high': 3, 'unknown': 0}
        if 'severity' in df_exploded.columns:
            df_exploded['severity_numeric'] = df_exploded['severity'].map(severity_map).fillna(0)
        else:
            df_exploded['severity_numeric'] = 0
            st.warning("No 'severity' column found in synthesized data")
        
        # Visualizations (top to bottom)
        if 'need_type' in df_exploded.columns:
            st.subheader("Distribution of Needs")
            need_counts = df_exploded['need_type'].value_counts().to_dict()
            if need_counts:
                fig_needs = px.bar(
                    x=list(need_counts.keys()),
                    y=list(need_counts.values()),
                    labels={"x": "Need Type", "y": "Count"},
                    title="Needs Distribution"
                )
                fig_needs.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_needs, use_container_width=True)
            else:
                st.info("No needs data available.")

        if 'severity' in df_exploded.columns:
            st.subheader("Distribution of Severity Levels")
            severity_counts = df_exploded['severity'].value_counts().to_dict()
            if severity_counts:
                fig_severity = px.bar(
                    x=list(severity_counts.keys()),
                    y=list(severity_counts.values()),
                    labels={"x": "Severity", "y": "Count"},
                    title="Severity Distribution"
                )
                fig_severity.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_severity, use_container_width=True)
            else:
                st.info("No severity data available.")

        st.subheader("Geographic Distribution (Heatmap & Points)")
        df_geo = pd.DataFrame(geocode_locations(df_exploded.to_dict('records'), country))
        if df_geo.empty:
            st.info("No geocoded locations available.")
        else:
            # Create figure with go.Figure for better control
            fig = go.Figure()

            # Add heatmap for severity intensity
            if 'severity_numeric' in df_geo.columns:
                fig.add_trace(
                    go.Densitymapbox(
                        lat=df_geo['latitude'],
                        lon=df_geo['longitude'],
                        z=df_geo['severity_numeric'],
                        radius=20,
                        colorscale="YlOrRd",
                        showscale=True,
                        colorbar=dict(title="Severity Intensity")
                    )
                )

            # Add scatter points for needs
            if 'need_type' in df_geo.columns:
                scatter_fig = px.scatter_map(
                    df_geo,
                    lat='latitude',
                    lon='longitude',
                    color='need_type',
                    size='severity_numeric',
                    hover_name='location',
                    hover_data={'severity': True, 'need_type': True} if 'severity' in df_geo.columns else {'need_type': True},
                    opacity=0.7
                )
                for trace in scatter_fig.data:
                    fig.add_trace(trace)

            # Update layout with MapLibre style
            fig.update_layout(
                mapbox=dict(
                    center=dict(lat=df_geo['latitude'].mean(), lon=df_geo['longitude'].mean()),
                    zoom=5,
                    style="https://tiles.stadiamaps.com/styles/outdoors.json"  # External MapLibre style
                ),
                margin={"r": 0, "t": 50, "l": 0, "b": 0},
                height=600,
                title="Geographic Distribution (Severity Heatmap & Need Points)"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Show raw synthesized data
        st.subheader("Structured Disaster Information")
        st.dataframe(df, use_container_width=True)

    if rumors:
        st.subheader("🚨 Identified Rumors")
        df_rumors = pd.DataFrame(rumors)
        st.dataframe(df_rumors, use_container_width=True)
        
        # Show rumor statistics
        if 'confidence' in df_rumors.columns:
            avg_confidence = df_rumors['confidence'].mean() if df_rumors['confidence'].notna().any() else 0
            st.metric("Average Rumor Detection Confidence", f"{avg_confidence:.2f}")
    elif synthesized:  # Only show this if we have synthesized data but no rumors
        st.info("✅ No rumors identified in the dataset.")

    if not synthesized and not rumors:
        st.info("📤 No data processed yet. Upload a dataset and run the pipeline to see results.")

    st.markdown("---")
    st.caption("🚀 Powered by Streamlit | Complete Disaster Response Pipeline (identical to main.py)")

if __name__ == "__main__":
    main()