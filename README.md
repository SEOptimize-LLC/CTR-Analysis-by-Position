# CTR Analysis by Position - All-in-One SEO Tool

**Comprehensive Streamlit application** that analyzes Google Search Console data, fetches search volume from DataForSEO, and calculates Click-Through Rate (CTR) by position to estimate organic traffic potential.

🔗 **Repository:** [github.com/SEOptimize-LLC/CTR-Analysis-by-Position](https://github.com/SEOptimize-LLC/CTR-Analysis-by-Position)

---

## 🎯 What This Tool Does

This app consolidates **3 separate SEO tools** into one unified workflow:

1. **Google Search Console Data Processing** - Upload and parse GSC performance reports
2. **Data Cleaning** - Automatic validation and cleaning of queries, URLs, and metrics
3. **Search Volume Integration** - Fetch US search volume and keyword difficulty from DataForSEO
4. **CTR Analysis** - Calculate average CTR by position (1-10) and estimate traffic potential

### Key Features

✅ **Upload GSC Data** - Supports CSV and Excel files with robust parsing
✅ **Auto-Clean Data** - Removes non-English queries, invalid URLs, special characters
✅ **Position Bucketing** - Groups queries by position (1.00-1.99 = position 1, etc.)
✅ **Search Volume API** - Integrates with DataForSEO Google Ads Search Volume
✅ **CTR Calculation** - Aggregates clicks/impressions by position bucket
✅ **Traffic Estimation** - Estimates monthly traffic: CTR × Search Volume
✅ **Dual Output** - Position aggregates + per-query details
✅ **Export Options** - Download results as CSV or Excel

---

## 📊 How It Works

### Workflow

```
Upload GSC Report → Clean Data → Bucket by Position (1-10) →
Fetch Search Volume (DataForSEO) → Calculate CTR by Position →
Estimate Traffic → Export Results
```

### Example Output

**Position Aggregates Table:**
| Position | Queries | Clicks | Impressions | CTR % | Search Volume | Est. Traffic |
|----------|---------|--------|-------------|-------|---------------|--------------|
| 1        | 45      | 1,250  | 15,000      | 8.33% | 125,000       | 10,417       |
| 2        | 128     | 980    | 12,500      | 7.84% | 234,000       | 18,346       |
| 3        | 203     | 745    | 10,800      | 6.90% | 198,000       | 13,662       |

**Per-Query Details Table:**
| Query | Position | Bucket | Clicks | Impressions | CTR % | Search Volume |
|-------|----------|--------|--------|-------------|-------|---------------|
| seo tips | 5.6 | 5 | 150 | 2,440 | 6.15% | 1,000 |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- DataForSEO API credentials ([Sign up here](https://app.dataforseo.com/))
- Google Search Console performance data export

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/SEOptimize-LLC/CTR-Analysis-by-Position.git
cd CTR-Analysis-by-Position
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure DataForSEO credentials:**
```bash
# Copy the template
cp .streamlit/secrets_template.toml .streamlit/secrets.toml

# Edit .streamlit/secrets.toml and add your credentials:
[dataforseo]
login = "your-dataforseo-email@example.com"
password = "your-dataforseo-api-password"
```

4. **Run the app:**
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📖 Usage Guide

### Step 1: Upload Data

1. Export your GSC performance data:
   - Go to Google Search Console → Performance
   - Set your date range (last 3 months recommended)
   - Click **Export** → Download as CSV or Excel

2. Upload the file in the app (Tab 1)
3. Verify that required columns are detected:
   - ✅ Query
   - ✅ Position
   - ✅ Clicks
   - ✅ Impressions

### Step 2: Clean Data

1. Navigate to **Tab 2: Clean Data**
2. Review which columns will be cleaned
3. Click **🧹 Clean Data**
4. Review cleaning statistics:
   - Rows removed (non-English, URLs, invalid data)
   - Retention rate
   - Per-column breakdown

### Step 3: Fetch Search Volume & Analyze

1. Navigate to **Tab 3: Fetch Search Volume**
2. Review position distribution (queries per position bucket)
3. Check unique query count and cost estimation
4. Click **🔍 Fetch Search Volume & Calculate CTR**
5. Wait for batch processing to complete (~1 second per 1000 keywords)

**Note:** Queries without search volume data will be **excluded** from calculations.

### Step 4: View Results

1. Navigate to **Tab 4: Results**
2. View summary metrics:
   - Total queries analyzed
   - Total search volume
   - Estimated monthly traffic
   - Weighted average CTR

3. **Download results:**
   - **Position Aggregates** - CTR by position (1-10) with traffic estimates
   - **Query Details** - Full data for every query with search volume

---

## 💰 API Costs

**DataForSEO Pricing:**
- Google Ads Search Volume: ~$0.0003 per keyword
- Example costs:
  - 500 queries = ~$0.15
  - 1,000 queries = ~$0.30
  - 5,000 queries = ~$1.50

The app shows cost estimates before making API calls.

---

## 🔧 Configuration

### Adjustable Settings

**In the Sidebar:**
- **Max Position:** Analyze positions 1-20 (default: 10)

**In the Code (streamlit_app.py):**
```python
LOCATION_CODE = 2840  # US (change for other countries)
LANGUAGE_CODE = "en"  # Language code
BATCH_SIZE = 1000     # Keywords per API request
MAX_POSITION = 10     # Default max position to analyze
```

### Country Codes

To analyze search volume for different countries, change `LOCATION_CODE`:
- 2840 = United States
- 2826 = United Kingdom
- 2124 = Canada
- [Full list of location codes](https://docs.dataforseo.com/v3/appendix/locations/)

---

## 📂 Project Structure

```
CTR Analysis by Position/
├── streamlit_app.py          # Main application (~900 lines)
├── requirements.txt           # Python dependencies
├── .streamlit/
│   ├── secrets.toml          # Your credentials (gitignored)
│   └── secrets_template.toml # Template for credentials
├── credentials/              # For future GSC OAuth (optional)
├── README.md                 # This file
└── .gitignore               # Git ignore rules
```

---

## 🧮 Calculation Methods

### Position Bucketing
```python
# Floor function: 1.00-1.99 → bucket 1, 5.6 → bucket 5
position_bucket = int(position)

# Only positions 1-10 are analyzed (configurable)
```

### CTR Calculation
```python
# Aggregate CTR per position (NOT weighted average)
position_ctr = sum(clicks) / sum(impressions)
```

### Traffic Estimation
```python
# Estimated monthly organic traffic
estimated_traffic = position_ctr × total_search_volume
```

---

## 🐛 Troubleshooting

### "Missing required columns" error
**Solution:** Ensure your GSC export includes: query, position, clicks, impressions

### "DataForSEO credentials not configured"
**Solution:** Create `.streamlit/secrets.toml` and add your API credentials (see Installation step 3)

### "No results returned from API"
**Possible causes:**
- Invalid API credentials
- Insufficient API balance
- Network connectivity issues

**Solution:** Check your DataForSEO account balance and credentials

### File parsing errors
**Solution:** Try these steps:
1. Save your file as UTF-8 encoded CSV
2. Use semicolon or comma as delimiter
3. Remove any special formatting from Excel before export

---

## 📊 Example Use Cases

### 1. Forecast Traffic from Ranking Improvements
**Scenario:** "If I improve these 50 keywords from position 5 to position 3, what traffic increase can I expect?"

**Solution:**
1. Upload GSC data
2. See current CTR for position 5 vs position 3
3. Calculate traffic difference: (CTR_pos3 - CTR_pos5) × Search Volume

### 2. Prioritize SEO Efforts
**Scenario:** "Which position has the highest traffic potential if I improve rankings?"

**Solution:**
1. Review Total Search Volume per position
2. Focus on positions with high volume and high CTR improvement potential
3. Example: Position 4 might have more volume than position 2

### 3. Estimate ROI of SEO Campaigns
**Scenario:** "What's the traffic value of improving rankings?"

**Solution:**
1. Calculate estimated traffic increase
2. Multiply by conversion rate × customer value
3. Compare to SEO campaign cost

---

## 🔄 Data Flow

```
┌─────────────────┐
│  GSC CSV/Excel  │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Parse & Detect │ (Auto-detect columns and delimiters)
│     Columns     │
└────────┬────────┘
         ↓
┌─────────────────┐
│   Clean Data    │ (Remove non-English, URLs, invalid values)
└────────┬────────┘
         ↓
┌─────────────────┐
│ Bucket Positions│ (floor(position), keep 1-10 only)
└────────┬────────┘
         ↓
┌─────────────────┐
│ Extract Unique  │ (Deduplicate queries, validate keywords)
│    Queries      │
└────────┬────────┘
         ↓
┌─────────────────┐
│ DataForSEO API  │ (Batch: 1000 keywords/request, 1s delay)
│  Search Volume  │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Merge & Filter  │ (EXCLUDE queries without search volume)
└────────┬────────┘
         ↓
┌─────────────────┐
│  Aggregate by   │ (CTR = sum(clicks)/sum(impressions))
│    Position     │ (Traffic = CTR × Search Volume)
└────────┬────────┘
         ↓
┌─────────────────┐
│ Display Results │ (2 tables: aggregates + details)
│  & Export       │ (CSV + Excel downloads)
└─────────────────┘
```

---

## 🤝 Contributing

This is a proprietary tool for SEOptimize LLC. For questions or issues:
- **Email:** [Your contact email]
- **Issues:** [GitHub Issues](https://github.com/SEOptimize-LLC/CTR-Analysis-by-Position/issues)

---

## 📝 License

Copyright © 2025 SEOptimize LLC. All rights reserved.

---

## 🙏 Acknowledgments

**Integrates code patterns from:**
- Google Search Console Connector
- Google Search Console Data Cleaner
- Fetch-Data-For-SEO-Monthly-Searches

**Built with:**
- [Streamlit](https://streamlit.io/) - Web framework
- [Pandas](https://pandas.pydata.org/) - Data processing
- [DataForSEO API](https://dataforseo.com/) - Search volume data

**Developed with:**
- [Claude Code](https://claude.ai/claude-code) by Anthropic

---

## 📞 Support

For DataForSEO API support:
- [Documentation](https://docs.dataforseo.com/)
- [Support Portal](https://dataforseo.com/support)

For Streamlit issues:
- [Streamlit Docs](https://docs.streamlit.io/)
- [Community Forum](https://discuss.streamlit.io/)

---

**Last Updated:** January 2025
**Version:** 1.0.0
**Author:** Roger SEO / SEOptimize LLC
