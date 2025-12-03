# Challenge H-001: Automated Insight Engine


A complete Python solution for automated marketing campaign performance analysis and report generation. This tool ingests CSV data, computes key performance indicators (KPIs), generates visualizations, and creates professional PowerPoint presentations with AI-powered executive summaries.

## 📋 Problem Statement (H-001)

The Automated Insight Engine addresses the need for rapid, data-driven insights from marketing campaign performance data. It automates the entire pipeline from raw CSV data to presentation-ready reports, eliminating manual data processing and visualization work.

## 📊 System Architecture Diagram

![System Architecture](./assets/diagram-export-3-12-2025-11_49_03-am.png)


**Key Requirements:**
- Ingest marketing campaign CSV data
- Compute comprehensive KPIs (Impressions, Clicks, Spend, Visits, CTR, CPC)
- Generate time series and comparative visualizations
- Create professional PPTX presentations
- AI-powered executive summary generation
- Modern, interactive Streamlit UI for configuration and preview

## 🛠️ Technology Stack

- **Python 3.11+** - Core language
- **pandas** - Data processing and analysis
- **matplotlib** - Chart generation
- **python-pptx** - PowerPoint presentation creation
- **Streamlit** - Interactive web UI
- **pytest** - Testing framework
- **Groq/OpenAI API** - LLM integration for executive summaries

**Optional Upgrades:**
- `polars` - High-performance data processing (mentioned in README as upgrade path)
- `libreoffice` - PDF conversion from PPTX

## 🎯 Approach

The solution follows a modular, pipeline-based architecture:

1. **Data Ingestion** - Load and validate CSV data with required columns
2. **Data Filtering** - Apply date range, campaign, location, and channel filters
3. **KPI Computation** - Calculate aggregated metrics (totals, averages, rates)
4. **Visualization** - Generate time series and bar charts as PNG images
5. **Summary Generation** - Use LLM (if available) or template-based executive summary
6. **Presentation Creation** - Assemble PPTX with title, summary, visuals, top campaigns, and appendix slides
7. **UI Interaction** - Streamlit interface for configuration, preview, and download

**Key Design Decisions:**
- **Modular Functions** - Each component (validation, KPIs, charts, LLM) is independently testable
- **Fallback Strategy** - LLM integration gracefully falls back to template summaries
- **Security First** - Only aggregated metrics sent to LLM APIs, no raw PII
- **Flexible Input** - Supports both uploaded CSVs and included sample data
- **Batch & Interactive** - Both CLI and Streamlit interfaces available

## 📁 Project Structure

```
.
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── .gitignore                                   # Git ignore rules
├── src/
│   ├── __init__.py
│   ├── auto_report.py                           # Core pipeline (ingest → transform → visualize → PPTX)
│   ├── utils.py                                 # Helper functions (validation, KPIs, charts, LLM wrapper)
│   └── app.py                                   # Streamlit UI application
├── sample_data/
│   └── marketing_campaign_performance_sample.csv # Sample dataset (180 rows, 30 days, 6 campaigns)
├── assets/
│   └── logo.png                                 # Placeholder logo
├── outputs/                                     # Generated reports directory
│   └── .gitkeep
└── tests/
    ├── __init__.py
    └── test_basic_pipeline.py                   # Smoke tests for KPI computation
```

## 🚀 How to Run

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd Groundtruth
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD):**
     ```cmd
     venv\Scripts\activate.bat
     ```
   - **Linux/Mac:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up API keys for LLM features (if using AI mode):**
   
   **Recommended: Use a `.env` file (easiest method):**
   
   1. Copy the example file:
      ```bash
      # On Windows PowerShell:
      Copy-Item .env.example .env
      
      # On macOS/Linux:
      cp .env.example .env
      ```
   
   2. Edit `.env` and add your API key:
      ```bash
      GROQ_API_KEY=your_groq_api_key_here
      ```
   
   **Alternative: Set environment variable:**
   
   **For Groq (recommended, free tier available):**
   - **macOS/Linux:**
     ```bash
     export GROQ_API_KEY="your_groq_api_key_here"
     ```
   - **Windows (PowerShell):**
     ```powershell
     setx GROQ_API_KEY "your_groq_api_key_here"
     ```
   - **Windows (CMD):**
     ```cmd
     set GROQ_API_KEY=your_groq_api_key_here
     ```
   
   **For OpenAI (alternative):**
   - **macOS/Linux:**
     ```bash
     export OPENAI_API_KEY="your_openai_api_key_here"
     ```
   - **Windows (PowerShell):**
     ```powershell
     setx OPENAI_API_KEY "your_openai_api_key_here"
     ```
   
   **Note:** The `.env` file method is recommended as it's easier and doesn't require restarting your terminal. The `.env` file is already in `.gitignore` so it won't be committed to git.

### Running the Streamlit UI

Launch the interactive web interface:

```bash
streamlit run src/app.py
```

The app will open in your default browser at `http://localhost:8501`.

**UI Features:**
- Upload CSV or use sample dataset
- Configure date range filters
- Select campaigns, locations, channels
- Preview KPIs and charts in real-time
- Choose Template or AI (Groq) summary mode
- **When `GROQ_API_KEY` is set, AI (Groq) is selected by default**
- Generate and download **PPTX and PDF** reports (PDF created automatically)

### Running Batch Script (Headless)

Generate a report directly from the command line:

```bash
# Basic usage with sample data
python src/auto_report.py --input sample_data/marketing_campaign_performance_sample.csv --out outputs/automated_report.pptx --no-llm

# With date filters
python src/auto_report.py --input sample_data/marketing_campaign_performance_sample.csv --out outputs/automated_report.pptx --date-start 2024-01-01 --date-end 2024-01-15 --no-llm

# With campaign filter
python src/auto_report.py --input sample_data/marketing_campaign_performance_sample.csv --out outputs/automated_report.pptx --campaigns "Summer Sale 2024" "Black Friday" --no-llm

# With Groq AI summary (requires GROQ_API_KEY)
python src/auto_report.py --input sample_data/marketing_campaign_performance_sample.csv --out outputs/automated_report.pptx --use-groq

# Clear LLM cache
python src/auto_report.py --clear-llm-cache
```

### Converting PPTX to PDF

PDF reports are generated **automatically** when you click \"Generate Report\" in the Streamlit UI:

- The app first tries **LibreOffice** (if installed) for best-quality conversion
- If LibreOffice is not available, it falls back to a **pure-Python pipeline** built with `reportlab` + `Pillow`
- If conversion succeeds, you'll see a **\"📄 Download PDF Report\"** button next to the PPTX download

If you prefer to run LibreOffice manually (optional):

```bash
libreoffice --headless --convert-to pdf outputs/automated_report.pptx --outdir outputs/
```

This will create `outputs/automated_report.pdf`.

### Running Tests

Execute the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run basic pipeline tests
pytest tests/test_basic_pipeline.py -v

# Run LLM integration tests (mocked, no network calls)
pytest tests/test_llm_integration.py -v

# Quick test run
pytest -q
```

**Note:** LLM integration tests use mocks and don't require API keys or network access.

### Testing the LLM Integration

After setting up your `GROQ_API_KEY`:

1. **Test in Streamlit:**
   - Run `streamlit run src/app.py`
   - Select "Sample Dataset"
   - Choose "AI (Groq)" mode
   - Click "Generate Report"
   - Verify you see "🤖 AI summary generated by Groq" with latency

2. **Test in Batch Mode:**
   ```bash
   python src/auto_report.py --input sample_data/marketing_campaign_performance_sample.csv --out outputs/test_groq.pptx --use-groq
   ```

3. **Test Fallback (Invalid Key):**
   ```bash
   # Temporarily set invalid key
   set GROQ_API_KEY=invalid_key
   python src/auto_report.py --input sample_data/marketing_campaign_performance_sample.csv --out outputs/test_fallback.pptx --use-groq
   # Should fall back to template summary
   ```

## 📊 Expected Output

The generated PowerPoint presentation (`automated_report.pptx`) contains:

1. **Title Slide**
   - Report title: "Marketing Campaign Performance Report"
   - Generation timestamp

2. **Executive Summary Slide**
   - Three key insights (bullet points)
   - Two actionable recommendations
   - Generated via LLM (if API key available) or template-based

3. **Key Visuals Slide**
   - **Left Chart:** Time series showing Visits and Impressions over time
   - **Right Chart:** Horizontal bar chart of top 5 campaigns by visits

4. **Top Campaigns Slide**
   - Bullet list of top 5 campaigns with:
     - Campaign name
     - Total visits
     - Total clicks
     - Total spend
     - CTR percentage

5. **Appendix Slide**
   - Complete KPI metrics:
     - Total Impressions
     - Total Clicks
     - Total Spend
     - Total Visits
     - Average CTR
     - Average CPC

**Sample KPIs from included dataset:**
- Total Impressions: ~2.7M
- Total Clicks: ~81K
- Total Spend: ~$120K
- Total Visits: ~70K
- Average CTR: ~3%
- Average CPC: ~$1.50

## 🎬 Demo Instructions

**For Hackathon Judges:**

1. **Start the Streamlit app:**
   ```bash
   streamlit run src/app.py
   ```

2. **Show the UI:**
   - Point out the modern layout with sidebar filters
   - Demonstrate KPI cards showing real-time metrics
   - Show the two-column chart layout

3. **Generate a report:**
   - Click "Generate Report" button
   - Show the progress spinner
   - Display the executive summary preview
   - Download the PPTX file

4. **Open the generated PPTX:**
   - Show all 5 slides
   - Highlight the professional formatting
   - Point out the AI-generated summary (if API key set) or template summary

5. **Demonstrate flexibility:**
   - Change date range filters and regenerate
   - Select specific campaigns and show filtered results
   - Switch between Quick and AI modes

6. **Show batch mode:**
   - Run the CLI command in terminal
   - Show the generated PPTX file

**Key Selling Points:**
- ✅ Complete end-to-end automation
- ✅ Modern, intuitive UI
- ✅ Professional presentation output
- ✅ Flexible filtering and configuration
- ✅ Robust error handling and validation
- ✅ AI integration with secure fallback

## 🔧 Tools & Libraries

### Core Dependencies
- `pandas>=2.0.0` - Data manipulation and analysis
- `matplotlib>=3.7.0` - Static chart generation
- `python-pptx>=0.6.21` - PowerPoint file creation
- `streamlit>=1.28.0` - Web UI framework
- `pytest>=7.4.0` - Testing framework

### Dependencies
- `openai>=1.3.0` - OpenAI API client (for LLM summaries)
- `groq>=0.4.0` - Groq API client (for free-tier LLM summaries)
- `python-dateutil>=2.8.2` - Date parsing utilities

### External Tools (Optional)
- `libreoffice` - For PPTX to PDF conversion

## 🔒 Security & Best Practices

- **No PII in API Calls:** Only aggregated metrics (totals, averages) are sent to LLM APIs
- **Environment Variables:** API keys stored in environment variables, never hardcoded
- **Graceful Fallbacks:** LLM failures automatically fall back to template summaries
- **Input Validation:** CSV data validated before processing
- **Error Handling:** Comprehensive error messages and logging throughout

## 🤖 LLM Integration (Groq)

The Automated Insight Engine includes AI-powered executive summary generation using Groq's fast LLM API.

### Enabling AI Mode

1. **Set the API Key:**
   - Get your free API key from [Groq Console](https://console.groq.com/)
   - Set it as an environment variable (see Installation section above)

2. **In Streamlit UI:**
   - Select "AI (Groq)" in the "Report Mode" section of the sidebar
   - Enable "Cache LLM Output" to avoid repeated API calls for identical contexts
   - Click "Generate Report" to create an AI-powered summary

3. **In Batch Mode:**
   ```bash
   python src/auto_report.py --input sample_data/marketing_campaign_performance_sample.csv --out outputs/automated_report.pptx --use-groq
   ```

### Security & Privacy

- **No PII Exported:** Only aggregated metrics (totals, averages, top campaigns) are sent to Groq
- **Sanitization:** The `sanitize_for_llm()` function explicitly removes any customer identifiers, emails, phone numbers, or raw user data
- **Context Limiting:** Input size is guarded (~3000 tokens max) and automatically truncated if needed
- **Safe Aggregates Only:** The system sends:
  - Time window (date range)
  - Total impressions, clicks, spend, visits
  - Average CTR, CPC, ROI
  - Top 3 campaigns (name, visits, CTR, spend only)

### Caching

- LLM responses are cached in `outputs/llm_cache.json` to avoid repeated API calls
- Cache is keyed by a hash of the aggregated context
- To clear the cache:
  ```bash
  python src/auto_report.py --clear-llm-cache
  ```
  Or manually delete `outputs/llm_cache.json`

### Fallback Behavior

- **No API Key:** Automatically uses template summary (no error)
- **API Failure:** Network errors, rate limits, or API errors fall back to template summary
- **Invalid Response:** Empty or malformed responses fall back to template summary
- **Timeout:** Requests timeout after 10 seconds, then fall back

### Performance

- **Latency:** Typically 1-3 seconds for Groq API calls
- **Retries:** Automatic exponential backoff with up to 2 retries for transient errors
- **Caching:** Identical contexts return cached responses instantly

### Default Behavior

- **Template Mode (Default):** Works without any API keys
- **AI Mode (Opt-in):** Must explicitly select "AI (Groq)" in UI or use `--use-groq` flag

## 🚀 Extension Ideas

1. **Enhanced Visualizations:**
   - Interactive Plotly charts in Streamlit
   - Heatmaps for campaign performance by date
   - Funnel charts for conversion analysis

2. **Advanced Analytics:**
   - Cohort analysis
   - Attribution modeling
   - Predictive forecasting

3. **Export Formats:**
   - Direct PDF generation (without LibreOffice)
   - HTML report export
   - Excel workbook with multiple sheets

4. **Data Sources:**
   - Database connectors (PostgreSQL, MySQL)
   - API integrations (Google Analytics, Facebook Ads)
   - Real-time data streaming

5. **Collaboration Features:**
   - Scheduled report generation
   - Email delivery
   - Cloud storage integration (S3, Google Drive)

6. **Performance Optimization:**
   - Migrate to `polars` for faster data processing
   - Caching for repeated queries
   - Parallel processing for large datasets

## 📝 Notes

- The default mode ("Quick") works without any API keys
- Sample data includes 180 rows covering 30 days and 6 campaigns
- All generated files are saved in the `outputs/` directory
- The Streamlit UI caches sample data for faster loading

## 📄 License

This project is created for hackathon challenge H-001.

## 👤 Author

Tharun Subramanian C
---

**Ready to generate insights? Run `streamlit run src/app.py` and start exploring!** 🚀

