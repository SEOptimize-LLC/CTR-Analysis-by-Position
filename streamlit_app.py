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
# DATA CLEANING FUNCTIONS (from Data Cleaner)
# ============================================================================

def detect_english_text(text):
    """
    Check if text contains primarily English characters
    Pattern from: Google Search Console Data Cleaner/gsc_data_cleaner.py lines 45-61
    """
    if pd.isna(text) or text == "":
        return False

    text = str(text)

    # Count Latin characters
    latin_chars = re.findall(r'[a-zA-Z\s]', text)
    total_chars = len(text.replace(' ', ''))

    if total_chars == 0:
        return False

    # At least 70% Latin characters
    latin_ratio = len(''.join(latin_chars).replace(' ', '')) / total_chars
    return latin_ratio > 0.7

def is_url(text):
    """Check if text is a URL"""
    if pd.isna(text):
        return False
    text = str(text).strip()
    return text.startswith(('http:', 'https:'))

def clean_query_column(series):
    """
    Clean query column - remove non-English, URLs, special chars
    Pattern from: Google Search Console Data Cleaner/gsc_data_cleaner.py lines 70-105
    """
    cleaned_series = series.copy()

    stats = {
        'original_count': len(cleaned_series),
        'non_english_removed': 0,
        'urls_removed': 0,
        'special_chars_cleaned': 0,
        'final_count': 0
    }

    # Remove non-English entries
    english_mask = cleaned_series.apply(detect_english_text)
    stats['non_english_removed'] = (~english_mask).sum()
    cleaned_series = cleaned_series[english_mask]

    # Remove URLs
    if len(cleaned_series) > 0:
        url_mask = cleaned_series.apply(lambda x: not is_url(x))
        stats['urls_removed'] = (~url_mask).sum()
        cleaned_series = cleaned_series[url_mask]

    # Clean special characters
    if len(cleaned_series) > 0:
        original_length = len(cleaned_series)
        cleaned_series = cleaned_series.apply(
            lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)) if pd.notna(x) else x
        )
        cleaned_series = cleaned_series.apply(
            lambda x: re.sub(r'\s+', ' ', str(x).strip()) if pd.notna(x) else x
        )
        # Remove empty strings
        cleaned_series = cleaned_series[cleaned_series.str.len() > 0]
        stats['special_chars_cleaned'] = original_length - len(cleaned_series)

    stats['final_count'] = len(cleaned_series)
    return cleaned_series, stats

def clean_page_column(series):
    """
    Clean page column - keep only HTTPS URLs
    Pattern from: Google Search Console Data Cleaner/gsc_data_cleaner.py lines 107-123
    """
    cleaned_series = series.copy()

    stats = {
        'original_count': len(cleaned_series),
        'non_https_removed': 0,
        'final_count': 0
    }

    # Keep only URLs starting with https:
    https_mask = cleaned_series.apply(
        lambda x: str(x).strip().startswith('https:') if pd.notna(x) else False
    )
    stats['non_https_removed'] = (~https_mask).sum()
    cleaned_series = cleaned_series[https_mask]
    stats['final_count'] = len(cleaned_series)

    return cleaned_series, stats

def clean_numeric_column(series):
    """
    Clean numeric columns - keep only valid numbers
    Pattern from: Google Search Console Data Cleaner/gsc_data_cleaner.py lines 125-144
    """
    cleaned_series = series.copy()

    stats = {
        'original_count': len(cleaned_series),
        'non_numeric_removed': 0,
        'final_count': 0
    }

    # Convert to numeric, coercing errors to NaN
    numeric_series = pd.to_numeric(cleaned_series, errors='coerce')

    # Remove NaN values
    non_null_mask = numeric_series.notna()
    stats['non_numeric_removed'] = (~non_null_mask).sum()
    cleaned_series = numeric_series[non_null_mask]
    stats['final_count'] = len(cleaned_series)

    return cleaned_series, stats

def process_dataframe(df, col_mapping):
    """
    Process entire dataframe with all cleaning operations
    Pattern from: Google Search Console Data Cleaner/gsc_data_cleaner.py lines 186-247
    """
    cleaning_stats = {}
    cleaned_df = df.copy()

    # Track valid indices
    original_indices = df.index.tolist()
    valid_indices = set(original_indices)

    # Clean query column
    if col_mapping['query']:
        cleaned_queries, query_stats = clean_query_column(df[col_mapping['query']])
        cleaning_stats[col_mapping['query']] = query_stats
        valid_indices = valid_indices.intersection(set(cleaned_queries.index))

    # Clean page column (optional)
    if col_mapping['page']:
        cleaned_pages, page_stats = clean_page_column(df[col_mapping['page']])
        cleaning_stats[col_mapping['page']] = page_stats
        valid_indices = valid_indices.intersection(set(cleaned_pages.index))

    # Clean position column
    if col_mapping['position']:
        cleaned_position, position_stats = clean_numeric_column(df[col_mapping['position']])
        cleaning_stats[col_mapping['position']] = position_stats
        valid_indices = valid_indices.intersection(set(cleaned_position.index))

    # Clean clicks and impressions
    for col_key in ['clicks', 'impressions']:
        if col_mapping[col_key]:
            cleaned_numeric, numeric_stats = clean_numeric_column(df[col_mapping[col_key]])
            cleaning_stats[col_mapping[col_key]] = numeric_stats
            valid_indices = valid_indices.intersection(set(cleaned_numeric.index))

    # Filter to valid rows
    final_indices = list(valid_indices)
    cleaned_df = df.loc[final_indices].copy()

    # Apply transformations to remaining rows
    if col_mapping['query'] and col_mapping['query'] in cleaned_df.columns:
        cleaned_df[col_mapping['query']] = cleaned_df[col_mapping['query']].apply(
            lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)) if pd.notna(x) else x
        )
        cleaned_df[col_mapping['query']] = cleaned_df[col_mapping['query']].apply(
            lambda x: re.sub(r'\s+', ' ', str(x).strip()) if pd.notna(x) else x
        )

    if col_mapping['position'] and col_mapping['position'] in cleaned_df.columns:
        cleaned_df[col_mapping['position']] = pd.to_numeric(
            cleaned_df[col_mapping['position']], errors='coerce'
        )

    for col_key in ['clicks', 'impressions']:
        if col_mapping[col_key] and col_mapping[col_key] in cleaned_df.columns:
            cleaned_df[col_mapping[col_key]] = pd.to_numeric(
                cleaned_df[col_mapping[col_key]], errors='coerce'
            )

    return cleaned_df, cleaning_stats

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
            st.markdown("""
            **Automatic data cleaning** removes:
            - Non-English queries
            - URLs in query column
            - Special characters
            - Non-HTTPS pages (if page column exists)
            - Invalid numeric values
            """)

            # Get column mapping
            col_mapping = identify_columns(st.session_state.raw_data)

            # Show what will be cleaned
            st.markdown("### Columns to Clean")
            col1, col2, col3 = st.columns(3)
            with col1:
                if col_mapping['query']:
                    st.success(f"✅ Query: {col_mapping['query']}")
                if col_mapping['position']:
                    st.success(f"✅ Position: {col_mapping['position']}")
            with col2:
                if col_mapping['clicks']:
                    st.success(f"✅ Clicks: {col_mapping['clicks']}")
                if col_mapping['impressions']:
                    st.success(f"✅ Impressions: {col_mapping['impressions']}")
            with col3:
                if col_mapping['page']:
                    st.info(f"ℹ️ Page: {col_mapping['page']} (optional)")
                else:
                    st.info("ℹ️ Page column not found (optional)")

            # Clean button
            if st.button("🧹 Clean Data", type="primary", use_container_width=True):
                with st.spinner("🧹 Cleaning data..."):
                    cleaned_df, cleaning_stats = process_dataframe(
                        st.session_state.raw_data,
                        col_mapping
                    )

                    st.session_state.cleaned_data = cleaned_df
                    st.session_state.cleaning_stats = cleaning_stats
                    st.session_state.current_step = 3

                st.success("✅ Data cleaned successfully!")

                # Show cleaning statistics
                st.markdown("### Cleaning Results")

                total_original = len(st.session_state.raw_data)
                total_cleaned = len(cleaned_df)
                retention_rate = (total_cleaned / total_original * 100) if total_original > 0 else 0

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Rows", f"{total_original:,}")
                with col2:
                    st.metric("Cleaned Rows", f"{total_cleaned:,}")
                with col3:
                    st.metric("Retention Rate", f"{retention_rate:.1f}%")

                # Detailed stats per column
                with st.expander("📊 Detailed Cleaning Stats"):
                    for col_name, stats in cleaning_stats.items():
                        st.markdown(f"**{col_name}**")
                        st.write(f"- Original: {stats['original_count']:,}")
                        st.write(f"- Final: {stats['final_count']:,}")
                        for key, value in stats.items():
                            if key not in ['original_count', 'final_count']:
                                st.write(f"- {key.replace('_', ' ').title()}: {value:,}")
                        st.markdown("---")

                st.info("👉 Continue to **Fetch Search Volume** tab")

            # Show cleaned data if available
            if st.session_state.cleaned_data is not None:
                st.markdown("### Cleaned Data Preview")
                st.dataframe(st.session_state.cleaned_data.head(10), use_container_width=True)

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
