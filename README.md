# Automated Insight Engine

**Challenge H-001: Automated Insight Engine**

A complete Python solution for automated marketing campaign performance analysis and report generation. This tool ingests CSV data, computes key performance indicators (KPIs), generates visualizations, and creates professional PowerPoint presentations with optional AI-powered executive summaries.

## 📋 Problem Statement (H-001)

The Automated Insight Engine addresses the need for rapid, data-driven insights from marketing campaign performance data. It automates the entire pipeline from raw CSV data to presentation-ready reports, eliminating manual data processing and visualization work.

**Key Requirements:**
- Ingest marketing campaign CSV data
- Compute comprehensive KPIs (Impressions, Clicks, Spend, Visits, CTR, CPC)
- Generate time series and comparative visualizations
- Create professional PPTX presentations
- Optional AI-powered executive summary generation
- Modern, interactive Streamlit UI for configuration and preview

## 🛠️ Technology Stack

- **Python 3.11+** - Core language
- **pandas** - Data processing and analysis
- **matplotlib** - Chart generation
- **python-pptx** - PowerPoint presentation creation
- **Streamlit** - Interactive web UI
- **pytest** - Testing framework
- **Groq/OpenAI API** - Optional LLM integration for executive summaries

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

5. **Optional: Set up API keys for LLM features (if using AI mode):**
   ```bash
   # For Groq (free tier available)
   set GROQ_API_KEY=your_groq_api_key_here
   
   # Or for OpenAI
   set OPENAI_API_KEY=your_openai_api_key_here
   ```

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
- Choose Quick (template) or AI (LLM) summary mode
- Generate and download PPTX reports

### Running Batch Script (Headless)

Generate a report directly from the command line:

```bash
# Basic usage with sample data
python src/auto_report.py --input sample_data/marketing_campaign_performance_sample.csv --out outputs/automated_report.pptx --no-llm

# With date filters
python src/auto_report.py --input sample_data/marketing_campaign_performance_sample.csv --out outputs/automated_report.pptx --date-start 2024-01-01 --date-end 2024-01-15 --no-llm

# With campaign filter
python src/auto_report.py --input sample_data/marketing_campaign_performance_sample.csv --out outputs/automated_report.pptx --campaigns "Summer Sale 2024" "Black Friday" --no-llm

# With LLM summary (requires API key)
python src/auto_report.py --input sample_data/marketing_campaign_performance_sample.csv --out outputs/automated_report.pptx
```

### Converting PPTX to PDF

If you have LibreOffice installed:

```bash
libreoffice --headless --convert-to pdf outputs/automated_report.pptx --outdir outputs/
```

This will create `outputs/automated_report.pdf`.

### Running Tests

Execute the test suite:

```bash
pytest tests/test_basic_pipeline.py -v
```

Or run with Python:

```bash
python -m pytest tests/test_basic_pipeline.py -v
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
- ✅ Optional AI integration with secure fallback

## 🔧 Tools & Libraries

### Core Dependencies
- `pandas>=2.0.0` - Data manipulation and analysis
- `matplotlib>=3.7.0` - Static chart generation
- `python-pptx>=0.6.21` - PowerPoint file creation
- `streamlit>=1.28.0` - Web UI framework
- `pytest>=7.4.0` - Testing framework

### Optional Dependencies
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

Created for Challenge H-001: Automated Insight Engine

---

**Ready to generate insights? Run `streamlit run src/app.py` and start exploring!** 🚀

