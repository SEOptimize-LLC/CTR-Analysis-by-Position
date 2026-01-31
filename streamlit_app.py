"""
CTR Analysis by Position - All-in-One SEO Tool

Integrates 3 existing tools into one comprehensive workflow:
1. Google Search Console Connector - OAuth & unlimited data fetching
2. Data Cleaner - GSC data validation and cleaning
3. DataForSEO Fetcher - Search volume and keyword difficulty
4. CTR Analysis (NEW) - Calculate CTR by position and estimate traffic

Author: Roger SEO / SEOptimize LLC
Built with Claude Code
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import base64
import time
import re
from io import BytesIO
from typing import Tuple, List, Dict, Optional
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

st.set_page_config(
    page_title="CTR Analysis by Position",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
LOCATION_CODE = 2840  # US only
LANGUAGE_CODE = "en"
BATCH_SIZE = 1000  # DataForSEO limit
MAX_POSITION = 10  # Only analyze 1-10
GOOGLE_ADS_API_ENDPOINT = "https://api.dataforseo.com/v3/keywords_data/google_ads/search_volume/live"

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""

    # Data storage
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    if 'bucketed_data' not in st.session_state:
        st.session_state.bucketed_data = None
    if 'search_volume_data' not in st.session_state:
        st.session_state.search_volume_data = None
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'detailed_df' not in st.session_state:
        st.session_state.detailed_df = None

    # Workflow state
    if 'data_source' not in st.session_state:
        st.session_state.data_source = 'upload'  # 'upload' or 'gsc'
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1  # 1-4
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

    # Cleaning stats
    if 'cleaning_stats' not in st.session_state:
        st.session_state.cleaning_stats = {}

# ============================================================================
# FILE UPLOAD & PARSING FUNCTIONS (from DataForSEO Fetcher & Data Cleaner)
# ============================================================================

def detect_delimiter(file):
    """Auto-detect CSV delimiter"""
    import csv
    file.seek(0)
    sample = file.read(8192)
    file.seek(0)

    if isinstance(sample, bytes):
        sample = sample.decode('utf-8', errors='ignore')

    try:
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter
        return delimiter
    except:
        # Fallback: try common delimiters
        for delim in [';', ',', '\t', '|']:
            if delim in sample:
                return delim
        return ','

def read_gsc_file(file):
    """
    Parse GSC CSV/Excel with robust error handling

    Tries multiple encodings and delimiters to handle various export formats.
    Pattern from: Fetch-Data-For-SEO-Monthly-Searches/app.py lines 71-208
    """
    file_extension = file.name.lower().split('.')[-1]

    try:
        if file_extension == 'csv':
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            delimiters = [',', '\t', ';', '|']

            df = None
            last_error = None

            # Strategy 1: Auto-detect delimiter
            for encoding in encodings:
                try:
                    file.seek(0)
                    df = pd.read_csv(
                        file,
                        encoding=encoding,
                        on_bad_lines='skip',
                        engine='python',
                        sep=None  # Auto-detect
                    )
                    if df is not None and not df.empty:
                        st.success(f"✅ File loaded with encoding: {encoding}")
                        return df
                except Exception as e:
                    last_error = e
                    continue

            # Strategy 2: Try explicit delimiters
            for encoding in encodings:
                for delimiter in delimiters:
                    try:
                        file.seek(0)
                        df = pd.read_csv(
                            file,
                            encoding=encoding,
                            delimiter=delimiter,
                            on_bad_lines='skip',
                            engine='python',
                            quoting=1
                        )
                        if df is not None and not df.empty and len(df.columns) > 1:
                            delimiter_name = {',': 'comma', '\t': 'tab',
                                            ';': 'semicolon', '|': 'pipe'}[delimiter]
                            st.success(f"✅ File loaded: encoding {encoding}, delimiter {delimiter_name}")
                            return df
                    except Exception as e:
                        last_error = e
                        continue

            if last_error:
                raise last_error

        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(file)
            st.success("✅ Excel file loaded successfully")
            return df
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None

    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.info("Try converting your file to CSV with UTF-8 encoding")
        return None

# ============================================================================
# COLUMN DETECTION FUNCTIONS (from Data Cleaner)
# ============================================================================

def identify_columns(df):
    """
    Identify GSC columns by checking variations

    Pattern from: Google Search Console Data Cleaner/gsc_data_cleaner.py lines 146-184
    """
    columns = df.columns.str.lower()

    # Query column variations
    query_variations = ['query', 'queries', 'keyword', 'keywords', 'search term']
    query_col = None
    for col in query_variations:
        matches = [c for c in df.columns if c.lower() == col]
        if matches:
            query_col = matches[0]
            break

    # Page column variations
    page_variations = ['page', 'landing page', 'address', 'url']
    page_col = None
    for col in page_variations:
        matches = [c for c in df.columns if c.lower() == col]
        if matches:
            page_col = matches[0]
            break

    # Position column variations
    position_variations = ['position', 'avg pos', 'avg position',
                          'avg. pos', 'avg. position', 'average position']
    position_col = None
    for col in position_variations:
        matches = [c for c in df.columns if c.lower() == col]
        if matches:
            position_col = matches[0]
            break

    # CTR column
    ctr_variations = ['ctr', 'click-through rate', 'click through rate']
    ctr_col = None
    for col in ctr_variations:
        matches = [c for c in df.columns if c.lower() == col]
        if matches:
            ctr_col = matches[0]
            break

    # Clicks column
    clicks_variations = ['clicks', 'click']
    clicks_col = None
    for col in clicks_variations:
        matches = [c for c in df.columns if c.lower() == col]
        if matches:
            clicks_col = matches[0]
            break

    # Impressions column
    impressions_variations = ['impressions', 'impression']
    impressions_col = None
    for col in impressions_variations:
        matches = [c for c in df.columns if c.lower() == col]
        if matches:
            impressions_col = matches[0]
            break

    return {
        'query': query_col,
        'page': page_col,
        'position': position_col,
        'ctr': ctr_col,
        'clicks': clicks_col,
        'impressions': impressions_col
    }

def validate_gsc_data(df, column_mapping):
    """Validate that required columns exist"""
    required = ['query', 'position', 'clicks', 'impressions']
    missing = [col for col in required if column_mapping[col] is None]

    if missing:
        st.error(f"❌ Missing required columns: {', '.join(missing)}")
        st.info("Required columns: query, position, clicks, impressions")
        return False

    return True

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application"""

    # Initialize session state
    init_session_state()

    # App header
    st.title("📊 CTR Analysis by Position")
    st.markdown("**All-in-One SEO Tool** | Upload GSC data → Clean → Fetch Search Volume → Analyze CTR by Position → Estimate Traffic")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Data source selection
        st.markdown("### Data Source")
        data_source = st.radio(
            "Choose data source:",
            options=["📤 Upload CSV/Excel", "🔗 Connect to GSC API (Coming Soon)"],
            index=0,
            disabled=True if st.session_state.raw_data is not None else False
        )

        st.session_state.data_source = 'upload' if '📤' in data_source else 'gsc'

        # Settings
        st.markdown("### Settings")
        max_position = st.slider(
            "Max Position",
            min_value=5,
            max_value=20,
            value=MAX_POSITION,
            step=1,
            help="Only analyze positions 1 to N"
        )

        # Progress indicator
        st.markdown("### Progress")
        steps = ["📤 Upload Data", "🧹 Clean Data", "🔍 Fetch Volume", "📊 Results"]
        for i, step in enumerate(steps, 1):
            if st.session_state.current_step > i:
                st.success(f"{step} ✅")
            elif st.session_state.current_step == i:
                st.info(f"{step} ⏳")
            else:
                st.text(f"{step} ⏸️")

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Data", "🧹 Clean Data", "🔍 Fetch Search Volume", "📊 Results"])

    with tab1:
        st.header("Step 1: Upload GSC Performance Report")
        st.markdown("Export your GSC performance data and upload it here. Supports CSV, Excel (.xlsx, .xls)")

        uploaded_file = st.file_uploader(
            "Choose your GSC performance report file",
            type=["csv", "xlsx", "xls"],
            help="Export from Google Search Console > Performance > Export"
        )

        if uploaded_file:
            with st.spinner("📖 Reading file..."):
                df = read_gsc_file(uploaded_file)

            if df is not None:
                st.session_state.raw_data = df
                st.session_state.current_step = 2

                st.success(f"✅ Loaded {len(df):,} rows")

                # Show preview
                st.markdown("### Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

                # Column detection
                col_mapping = identify_columns(df)

                st.markdown("### Detected Columns")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Query", col_mapping['query'] or "❌ Not found")
                    st.metric("Position", col_mapping['position'] or "❌ Not found")
                with col2:
                    st.metric("Clicks", col_mapping['clicks'] or "❌ Not found")
                    st.metric("Impressions", col_mapping['impressions'] or "❌ Not found")
                with col3:
                    st.metric("CTR", col_mapping['ctr'] or "⚠️ Optional")
                    st.metric("Page", col_mapping['page'] or "⚠️ Optional")

                # Validate
                if validate_gsc_data(df, col_mapping):
                    st.success("✅ All required columns detected!")
                    st.info("👉 Continue to **Clean Data** tab")
                else:
                    st.error("❌ Missing required columns. Please upload a valid GSC performance report.")

    with tab2:
        st.header("Step 2: Clean Data")
        if st.session_state.raw_data is None:
            st.warning("⚠️ Please upload data in Step 1 first")
        else:
            st.markdown("Data cleaning will be implemented in the next iteration")
            st.info("For now, proceeding with raw data...")
            # TODO: Implement cleaning functions

    with tab3:
        st.header("Step 3: Fetch Search Volume")
        if st.session_state.raw_data is None:
            st.warning("⚠️ Please upload and clean data first")
        else:
            st.markdown("Search volume fetching will be implemented in the next iteration")
            # TODO: Implement DataForSEO integration

    with tab4:
        st.header("Step 4: Results")
        if not st.session_state.processing_complete:
            st.warning("⚠️ Please complete previous steps first")
        else:
            st.markdown("Results display will be implemented in the next iteration")
            # TODO: Implement results display

if __name__ == "__main__":
    main()
