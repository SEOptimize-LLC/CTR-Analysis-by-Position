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
# POSITION BUCKETING FUNCTIONS (NEW)
# ============================================================================

def bucket_by_position(df, col_mapping, max_position=10):
    """
    Add position_bucket column using floor(position)
    Only keep positions 1-10 (1.00-1.99 = bucket 1, etc.)
    """
    bucketed_df = df.copy()

    position_col = col_mapping['position']
    if not position_col:
        st.error("Position column not found")
        return None

    # Ensure position is numeric
    bucketed_df[position_col] = pd.to_numeric(bucketed_df[position_col], errors='coerce')

    # Drop rows with NaN positions
    bucketed_df = bucketed_df[bucketed_df[position_col].notna()]

    # Create position bucket (floor function)
    bucketed_df['position_bucket'] = bucketed_df[position_col].apply(lambda x: int(x))

    # Filter to max_position
    bucketed_df = bucketed_df[bucketed_df['position_bucket'] <= max_position]
    bucketed_df = bucketed_df[bucketed_df['position_bucket'] >= 1]

    return bucketed_df

# ============================================================================
# DATAFORSEO API FUNCTIONS (from DataForSEO Fetcher)
# ============================================================================

def get_dataforseo_credentials():
    """Retrieve DataForSEO credentials from secrets"""
    try:
        login = st.secrets["dataforseo"]["login"]
        password = st.secrets["dataforseo"]["password"]
        return login, password
    except Exception:
        st.error("❌ DataForSEO credentials not configured in secrets.toml")
        st.info("""
        Add credentials to .streamlit/secrets.toml:

        [dataforseo]
        login = "your-email@example.com"
        password = "your-api-password"
        """)
        return None, None

def validate_and_clean_keywords(keywords_list, max_words=10):
    """
    Validate and clean keywords before API call
    Pattern from: Fetch-Data-For-SEO-Monthly-Searches/app.py lines 211-278
    """
    valid_keywords = []
    skipped_keywords = []
    duplicate_keywords = []
    seen_keywords = set()

    for keyword in keywords_list:
        original_keyword = keyword
        keyword = keyword.strip()

        if not keyword:
            skipped_keywords.append({'keyword': original_keyword, 'reason': 'Empty keyword'})
            continue

        # Remove invalid characters
        cleaned_keyword = re.sub(r'[^\w\s-]', '', keyword)
        cleaned_keyword = ' '.join(cleaned_keyword.split())

        if not cleaned_keyword:
            skipped_keywords.append({'keyword': original_keyword, 'reason': 'Only invalid characters'})
            continue

        # Check word count
        word_count = len(cleaned_keyword.split())
        if word_count > max_words:
            skipped_keywords.append({'keyword': original_keyword, 'reason': f'Too many words ({word_count})'})
            continue

        # Check duplicates
        cleaned_lower = cleaned_keyword.lower()
        if cleaned_lower in seen_keywords:
            duplicate_keywords.append({'keyword': original_keyword, 'reason': 'Duplicate'})
            continue

        seen_keywords.add(cleaned_lower)
        valid_keywords.append({
            'original': original_keyword,
            'cleaned': cleaned_keyword,
            'modified': cleaned_keyword.lower() != original_keyword.lower()
        })

    return valid_keywords, skipped_keywords, duplicate_keywords

def call_google_ads_api(keywords, login, password):
    """
    Call DataForSEO Google Ads Search Volume API
    Pattern from: Fetch-Data-For-SEO-Monthly-Searches/app.py lines 281-308
    """
    cred = f"{login}:{password}"
    encoded_cred = base64.b64encode(cred.encode('ascii')).decode('ascii')

    headers = {
        'Authorization': f'Basic {encoded_cred}',
        'Content-Type': 'application/json'
    }

    post_data = [{
        "keywords": keywords,
        "location_code": LOCATION_CODE,
        "language_code": LANGUAGE_CODE
    }]

    try:
        response = requests.post(GOOGLE_ADS_API_ENDPOINT, headers=headers, json=post_data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Google Ads API Error: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Response: {e.response.text}")
        return None

def get_latest_monthly_search_volume(monthly_searches):
    """
    Extract most recent month's search volume
    Pattern from: Fetch-Data-For-SEO-Monthly-Searches/app.py lines 311-328
    """
    if not monthly_searches:
        return 0

    sorted_months = sorted(
        monthly_searches,
        key=lambda x: (x.get('year', 0), x.get('month', 0)),
        reverse=True
    )

    if sorted_months:
        return sorted_months[0].get('search_volume', 0) or 0
    return 0

def process_google_ads_response(response_data):
    """
    Process API response to extract search volume
    Pattern from: Fetch-Data-For-SEO-Monthly-Searches/app.py lines 331-385
    """
    if not response_data or 'tasks' not in response_data:
        return None

    results = []
    for task in response_data.get('tasks', []):
        task_status = task.get('status_code')
        if task_status == 20000:  # Success
            task_result = task.get('result', [])
            for item in task_result:
                keyword = item.get('keyword', '')
                monthly_searches = item.get('monthly_searches', [])
                last_month_volume = get_latest_monthly_search_volume(monthly_searches)
                competition = item.get('competition', 'N/A')

                results.append({
                    'Keyword': keyword,
                    'US Search Volume (Last Month)': last_month_volume,
                    'Keyword Difficulty': competition
                })
        else:
            st.warning(f"Task failed with status code: {task_status}")

    if results:
        return pd.DataFrame(results)
    return None

def fetch_search_volume_batch(queries, login, password):
    """
    Batch fetch search volume with progress tracking
    Pattern from: Fetch-Data-For-SEO-Monthly-Searches/app.py lines 483-509
    """
    total_queries = len(queries)
    num_batches = (total_queries + BATCH_SIZE - 1) // BATCH_SIZE

    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(0, total_queries, BATCH_SIZE):
        batch = queries[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        status_text.text(f"Batch {batch_num}/{num_batches}: Fetching {len(batch)} keywords...")

        response = call_google_ads_api(batch, login, password)

        if response:
            batch_df = process_google_ads_response(response)
            if batch_df is not None:
                all_results.append(batch_df)

        # Update progress
        progress = min((i + BATCH_SIZE) / total_queries, 1.0)
        progress_bar.progress(progress)

        # Rate limiting (1 second between batches)
        if i + BATCH_SIZE < total_queries:
            time.sleep(1)

    progress_bar.progress(1.0)
    status_text.text(f"✅ Completed {num_batches} batches")

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        st.error("No results returned from API")
        return None

# ============================================================================
# CTR AGGREGATION FUNCTIONS (NEW)
# ============================================================================

def merge_and_filter_data(gsc_df, sv_df, col_mapping):
    """
    Merge GSC data with search volume and filter out queries without SV
    User requirement: EXCLUDE queries with missing search volume
    """
    query_col = col_mapping['query']

    # Merge on query column
    merged_df = gsc_df.merge(
        sv_df,
        left_on=query_col,
        right_on='Keyword',
        how='left'
    )

    # Track exclusions
    before_count = len(merged_df)

    # EXCLUDE queries with missing or zero search volume
    merged_df = merged_df[merged_df['US Search Volume (Last Month)'].notna()]
    merged_df = merged_df[merged_df['US Search Volume (Last Month)'] > 0]

    after_count = len(merged_df)
    excluded_count = before_count - after_count

    if excluded_count > 0:
        st.warning(f"⚠️ Excluded {excluded_count:,} queries with no search volume data")

    return merged_df, excluded_count

def aggregate_by_position(merged_df, col_mapping):
    """
    Aggregate metrics by position bucket
    CTR = sum(clicks) / sum(impressions) per bucket
    """
    position_col = col_mapping['position']
    clicks_col = col_mapping['clicks']
    impressions_col = col_mapping['impressions']

    agg_results = []

    for position in range(1, 11):  # Positions 1-10
        bucket_data = merged_df[merged_df['position_bucket'] == position]

        if len(bucket_data) == 0:
            continue

        # Calculate metrics
        total_queries = len(bucket_data)
        total_clicks = bucket_data[clicks_col].sum()
        total_impressions = bucket_data[impressions_col].sum()
        position_ctr = total_clicks / total_impressions if total_impressions > 0 else 0
        total_search_volume = bucket_data['US Search Volume (Last Month)'].sum()
        estimated_traffic = position_ctr * total_search_volume

        agg_results.append({
            'Position': position,
            'Total Queries': total_queries,
            'Total Clicks': int(total_clicks),
            'Total Impressions': int(total_impressions),
            'Position CTR': position_ctr,
            'CTR %': f"{position_ctr * 100:.2f}%",
            'Total Search Volume': int(total_search_volume),
            'Estimated Monthly Traffic': int(estimated_traffic)
        })

    return pd.DataFrame(agg_results)

def convert_df_to_excel(df):
    """
    Convert dataframe to Excel with formatting
    Pattern from: Fetch-Data-For-SEO-Monthly-Searches/app.py lines 388-400
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')

        # Auto-adjust column width
        worksheet = writer.sheets['Results']
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).apply(len).max(), len(col)) + 2
            worksheet.set_column(i, i, min(max_len, 50))

    return output.getvalue()

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
        st.header("Step 3: Fetch Search Volume & Analyze")
        if st.session_state.cleaned_data is None:
            st.warning("⚠️ Please upload and clean data first (Tabs 1 & 2)")
        else:
            col_mapping = identify_columns(st.session_state.cleaned_data)

            # Step 3.1: Position bucketing
            st.markdown("### Step 3.1: Position Bucketing")
            st.info(f"Filtering to positions 1-{max_position} only")

            if st.session_state.bucketed_data is None:
                bucketed_df = bucket_by_position(
                    st.session_state.cleaned_data,
                    col_mapping,
                    max_position
                )

                if bucketed_df is not None:
                    st.session_state.bucketed_data = bucketed_df

                    # Show bucketing stats
                    original_count = len(st.session_state.cleaned_data)
                    bucketed_count = len(bucketed_df)
                    excluded = original_count - bucketed_count

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("After Cleaning", f"{original_count:,}")
                    with col2:
                        st.metric(f"Positions 1-{max_position}", f"{bucketed_count:,}")
                    with col3:
                        st.metric("Excluded (>10)", f"{excluded:,}")

                    # Position distribution
                    st.markdown("#### Position Distribution")
                    position_counts = bucketed_df['position_bucket'].value_counts().sort_index()
                    st.bar_chart(position_counts)

            # Step 3.2: Extract unique queries
            if st.session_state.bucketed_data is not None:
                st.markdown("### Step 3.2: Extract Unique Queries")

                query_col = col_mapping['query']
                unique_queries = st.session_state.bucketed_data[query_col].unique().tolist()

                st.info(f"Found **{len(unique_queries):,}** unique queries")

                # Validate and clean keywords
                valid, skipped, duplicates = validate_and_clean_keywords(unique_queries)
                clean_queries = [kw['cleaned'] for kw in valid]

                if len(skipped) > 0:
                    st.warning(f"Skipped {len(skipped)} invalid keywords")
                if len(duplicates) > 0:
                    st.info(f"Removed {len(duplicates)} duplicates")

                st.success(f"✅ {len(clean_queries):,} keywords ready for API")

                # Cost estimation
                estimated_cost = len(clean_queries) * 0.0003
                st.markdown("### 💰 Cost Estimation")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Keywords to Fetch", f"{len(clean_queries):,}")
                with col2:
                    st.metric("Estimated Cost", f"${estimated_cost:.2f}")

                # Fetch button
                if st.button("🔍 Fetch Search Volume & Calculate CTR", type="primary", use_container_width=True):
                    login, password = get_dataforseo_credentials()

                    if login and password:
                        with st.spinner("📊 Fetching search volume data from DataForSEO..."):
                            sv_df = fetch_search_volume_batch(clean_queries, login, password)

                        if sv_df is not None:
                            st.session_state.search_volume_data = sv_df
                            st.success(f"✅ Fetched search volume for {len(sv_df):,} keywords")

                            # Merge and filter
                            st.markdown("### Step 3.3: Merge & Filter Data")
                            merged_df, excluded_count = merge_and_filter_data(
                                st.session_state.bucketed_data,
                                sv_df,
                                col_mapping
                            )

                            if len(merged_df) > 0:
                                # Aggregate by position
                                st.markdown("### Step 3.4: Calculate CTR by Position")
                                results_df = aggregate_by_position(merged_df, col_mapping)

                                st.session_state.results_df = results_df
                                st.session_state.detailed_df = merged_df
                                st.session_state.processing_complete = True
                                st.session_state.current_step = 4

                                st.success("✅ Analysis complete!")
                                st.balloons()
                                st.info("👉 View results in the **Results** tab")
                            else:
                                st.error("No data left after filtering. All queries had missing search volume.")
                        else:
                            st.error("Failed to fetch search volume data")

            # Show search volume data if available
            if st.session_state.search_volume_data is not None:
                st.markdown("### Search Volume Data Preview")
                st.dataframe(st.session_state.search_volume_data.head(10), use_container_width=True)

    with tab4:
        st.header("Step 4: Results")
        if not st.session_state.processing_complete:
            st.warning("⚠️ Please complete analysis in Tab 3 first")
        else:
            results_df = st.session_state.results_df
            detailed_df = st.session_state.detailed_df

            # Summary metrics
            st.markdown("### 📊 Summary Metrics")

            total_queries = results_df['Total Queries'].sum()
            total_volume = results_df['Total Search Volume'].sum()
            total_estimated_traffic = results_df['Estimated Monthly Traffic'].sum()
            weighted_avg_ctr = results_df['Total Clicks'].sum() / results_df['Total Impressions'].sum()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Queries Analyzed", f"{total_queries:,}")
            with col2:
                st.metric("Total Search Volume", f"{total_volume:,.0f}")
            with col3:
                st.metric("Est. Monthly Traffic", f"{total_estimated_traffic:,.0f}")
            with col4:
                st.metric("Weighted Avg CTR", f"{weighted_avg_ctr * 100:.2f}%")

            st.markdown("---")

            # Table 1: Position Aggregates
            st.markdown("### 📈 Table 1: CTR Analysis by Position")
            st.markdown("*This shows the aggregate CTR and estimated traffic for each position bucket (1-10)*")

            # Format for display
            display_df = results_df.copy()

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Position": st.column_config.NumberColumn("Position", help="Position bucket (1-10)"),
                    "Total Queries": st.column_config.NumberColumn("Queries", format="%d"),
                    "Total Clicks": st.column_config.NumberColumn("Clicks", format="%d"),
                    "Total Impressions": st.column_config.NumberColumn("Impressions", format="%d"),
                    "Position CTR": st.column_config.NumberColumn("CTR (Decimal)", format="%.4f"),
                    "CTR %": st.column_config.TextColumn("CTR %"),
                    "Total Search Volume": st.column_config.NumberColumn("Search Volume", format="%d"),
                    "Estimated Monthly Traffic": st.column_config.NumberColumn("Est. Traffic", format="%d"),
                }
            )

            # Download buttons for position aggregates
            col1, col2 = st.columns(2)
            with col1:
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Position Aggregates (CSV)",
                    data=csv,
                    file_name=f"ctr_by_position_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col2:
                excel = convert_df_to_excel(results_df)
                st.download_button(
                    label="📥 Download Position Aggregates (Excel)",
                    data=excel,
                    file_name=f"ctr_by_position_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            st.markdown("---")

            # Table 2: Per-Query Details
            st.markdown("### 🔍 Table 2: Per-Query Details")
            st.markdown(f"*Detailed data for all {len(detailed_df):,} queries with search volume*")

            col_mapping = identify_columns(st.session_state.raw_data)
            query_col = col_mapping['query']
            position_col = col_mapping['position']
            clicks_col = col_mapping['clicks']
            impressions_col = col_mapping['impressions']

            # Select relevant columns for display
            detail_display_cols = [
                query_col,
                position_col,
                'position_bucket',
                clicks_col,
                impressions_col,
                'US Search Volume (Last Month)',
                'Keyword Difficulty'
            ]

            # Filter to only these columns
            detail_display = detailed_df[detail_display_cols].copy()

            # Add CTR column
            detail_display['CTR %'] = (detail_display[clicks_col] / detail_display[impressions_col] * 100).round(2).astype(str) + '%'

            # Show first 100 rows by default
            st.dataframe(
                detail_display.head(100),
                use_container_width=True,
                hide_index=True
            )

            if len(detail_display) > 100:
                st.info(f"Showing first 100 of {len(detail_display):,} total queries. Download full data below.")

            # Download buttons for detailed data
            col1, col2 = st.columns(2)
            with col1:
                csv_detail = detail_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download All Query Details (CSV)",
                    data=csv_detail,
                    file_name=f"query_details_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col2:
                excel_detail = convert_df_to_excel(detail_display)
                st.download_button(
                    label="📥 Download All Query Details (Excel)",
                    data=excel_detail,
                    file_name=f"query_details_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            st.markdown("---")
            st.success("🎉 Analysis complete! You can download both the position aggregates and detailed query data above.")

if __name__ == "__main__":
    main()
