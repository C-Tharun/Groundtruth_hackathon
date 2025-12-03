"""
Streamlit UI for Automated Insight Engine.
Upload dataset, configure filters, preview KPIs and charts, generate reports.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from io import BytesIO

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logging.info(f"Loaded .env file from {env_path}")
except ImportError:
    # python-dotenv not installed, will use system environment variables only
    pass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    validate_csv_data,
    compute_kpis,
    create_time_series_chart,
    create_top_campaigns_chart,
    get_top_campaigns_list
)
from src.auto_report import generate_report

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Automated Insight Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1E88E5;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Fix KPI metric value color - make it dark and visible */
    [data-testid="stMetricValue"] {
        color: #262730 !important;
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        color: #505050 !important;
    }
    /* Ensure metric containers have proper contrast */
    div[data-testid="stMetric"] {
        color: #262730 !important;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_sample_data():
    """Load sample data with caching."""
    sample_path = Path(__file__).parent.parent / "sample_data" / "marketing_campaign_performance_sample.csv"
    if sample_path.exists():
        return pd.read_csv(sample_path)
    return None


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">📊 Automated Insight Engine</div>', unsafe_allow_html=True)
    
    # Top bar with timestamp and logo placeholder
    col_header1, col_header2, col_header3 = st.columns([2, 1, 1])
    with col_header1:
        st.markdown("**Challenge H-001: Automated Insight Engine**")
    with col_header3:
        st.markdown(f"*{datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data source selection
        st.subheader("Data Source")
        data_source = st.radio(
            "Choose data source:",
            ["Sample Dataset", "Upload CSV"],
            index=0
        )
        
        df = None
        if data_source == "Sample Dataset":
            df = load_sample_data()
            if df is None:
                st.error("Sample data not found. Please upload a CSV file.")
                st.stop()
            else:
                st.success(f"✓ Loaded {len(df)} rows from sample data")
        else:
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            
            # Show CSV format requirements
            with st.expander("📋 Required CSV Format", expanded=False):
                st.markdown("""
                **Required Columns:**
                - `Date` - Date of the campaign (YYYY-MM-DD format)
                - `Campaign` - Campaign name
                - `Impressions` - Number of impressions (numeric)
                - `Clicks` - Number of clicks (numeric)
                - `Spend` - Amount spent (numeric)
                - `Visits` - Number of visits (numeric)
                
                **Optional Columns:**
                - `Location` - Location filter
                - `Channel` - Channel filter
                
                **Example:**
                ```csv
                Date,Campaign,Impressions,Clicks,Spend,Visits
                2024-01-01,Summer Sale,10000,500,1000.50,450
                2024-01-02,Black Friday,15000,750,1500.75,680
                ```
                """)
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success(f"✓ Loaded {len(df)} rows")
            else:
                st.info("👆 Please upload a CSV file to get started")
                st.stop()
        
        # Validate data
        if df is not None:
            is_valid, error_msg = validate_csv_data(df)
            if not is_valid:
                st.error(f"Data validation failed: {error_msg}")
                st.stop()
        
        st.divider()
        
        # Filters
        st.subheader("📅 Date Range")
        if df is not None and 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            min_date = df['Date'].min().date()
            max_date = df['Date'].max().date()
            
            date_range = st.date_input(
                "Select date range:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if isinstance(date_range, tuple) and len(date_range) == 2:
                date_start = date_range[0].strftime('%Y-%m-%d')
                date_end = date_range[1].strftime('%Y-%m-%d')
            else:
                date_start = None
                date_end = None
        else:
            date_start = None
            date_end = None
            st.warning("Date column not found")
        
        st.divider()
        
        # Campaign filter
        st.subheader("🎯 Campaign Filter")
        if df is not None and 'Campaign' in df.columns:
            all_campaigns = sorted(df['Campaign'].unique().tolist())
            selected_campaigns = st.multiselect(
                "Select campaigns (leave empty for all):",
                all_campaigns,
                default=[]
            )
        else:
            selected_campaigns = []
        
        # Location filter (if exists)
        selected_locations = []
        if df is not None and 'Location' in df.columns:
            st.subheader("📍 Location Filter")
            all_locations = sorted(df['Location'].unique().tolist())
            selected_locations = st.multiselect(
                "Select locations (leave empty for all):",
                all_locations,
                default=[]
            )
        
        # Channel filter (if exists)
        selected_channels = []
        if df is not None and 'Channel' in df.columns:
            st.subheader("📺 Channel Filter")
            all_channels = sorted(df['Channel'].unique().tolist())
            selected_channels = st.multiselect(
                "Select channels (leave empty for all):",
                all_channels,
                default=[]
            )
        
        st.divider()
        
        # Report generation mode
        st.subheader("🤖 Report Mode")
        # Check if API key is available to determine default
        groq_key_available = bool(os.getenv('GROQ_API_KEY'))
        default_index = 1 if groq_key_available else 0  # Default to AI if key available
        
        ai_mode = st.radio(
            "Summary generation:",
            ["Off (Template)", "AI (Groq)"],
            index=default_index,
            help="Template mode uses deterministic summary. AI mode uses Groq LLM (requires GROQ_API_KEY)."
        )
        use_groq = ai_mode == "AI (Groq)"
        
        # Cache option
        use_cache = st.checkbox(
            "Cache LLM Output",
            value=True,
            help="Cache Groq responses to avoid repeated API calls for identical contexts."
        )
        
        # Check for API key if AI mode selected
        groq_api_key_set = bool(os.getenv('GROQ_API_KEY'))
        
        # Debug: Show API key status
        if use_groq:
            api_key = os.getenv('GROQ_API_KEY')
            env_file_exists = (Path(__file__).parent.parent / '.env').exists()
            
            if api_key:
                source = "from .env file" if env_file_exists else "from environment variable"
                st.success(f"✓ GROQ_API_KEY detected {source} (length: {len(api_key)} chars)")
            else:
                st.error("⚠️ GROQ_API_KEY not set. Please set it to use AI mode.")
                st.info("**Option 1: Create a `.env` file in the project root:**")
                st.code("GROQ_API_KEY=your_groq_api_key_here", language="text")
                st.info("**Option 2: Set environment variable in PowerShell:**")
                st.code("$env:GROQ_API_KEY = 'your_key'", language="powershell")
                use_groq = False  # Disable AI mode if no key
    
    # Main content area
    if df is None:
        st.info("👈 Please configure data source in the sidebar")
        return
    
    # Apply filters
    df_filtered = df.copy()
    
    if date_start and date_end:
        df_filtered = df_filtered[
            (df_filtered['Date'] >= pd.to_datetime(date_start)) &
            (df_filtered['Date'] <= pd.to_datetime(date_end))
        ]
    
    if selected_campaigns:
        df_filtered = df_filtered[df_filtered['Campaign'].isin(selected_campaigns)]
    
    if selected_locations and 'Location' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Location'].isin(selected_locations)]
    
    if selected_channels and 'Channel' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Channel'].isin(selected_channels)]
    
    if df_filtered.empty:
        st.error("No data remaining after applying filters. Please adjust your filters.")
        return
    
    # Compute KPIs
    kpis = compute_kpis(df_filtered)
    top_campaigns = get_top_campaigns_list(df_filtered, top_n=5)
    
    # Display KPIs
    st.header("📈 Key Performance Indicators")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Impressions", f"{kpis['total_impressions']:,.0f}")
    with col2:
        st.metric("Clicks", f"{kpis['total_clicks']:,.0f}")
    with col3:
        st.metric("Spend", f"${kpis['total_spend']:,.2f}")
    with col4:
        st.metric("Visits", f"{kpis['total_visits']:,.0f}")
    with col5:
        st.metric("Avg CTR", f"{kpis['avg_ctr']:.2f}%")
    with col6:
        st.metric("Avg CPC", f"${kpis['avg_cpc']:.2f}")
    
    st.divider()
    
    # Charts (two-column layout)
    st.header("📊 Visualizations")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Performance Over Time")
        chart1_path = "outputs/temp_time_series.png"
        os.makedirs("outputs", exist_ok=True)
        create_time_series_chart(df_filtered, chart1_path)
        st.image(chart1_path)
    
    with col_chart2:
        st.subheader("Top 5 Campaigns by Visits")
        chart2_path = "outputs/temp_top_campaigns.png"
        create_top_campaigns_chart(df_filtered, chart2_path, top_n=5)
        st.image(chart2_path)
    
    st.divider()
    
    # Report generation
    st.header("📄 Generate Report")
    
    col_gen1, col_gen2 = st.columns([3, 1])
    with col_gen1:
        st.info("Click the button below to generate a PowerPoint presentation and PDF report with all insights.")
    
    with col_gen2:
        generate_button = st.button("🚀 Generate Report", type="primary", use_container_width=True)
    
    if generate_button:
        with st.spinner("Generating report... This may take a moment."):
            try:
                # Prepare filters
                campaigns_list = selected_campaigns if selected_campaigns else None
                locations_list = selected_locations if selected_locations else None
                channels_list = selected_channels if selected_channels else None
                
                # Determine input CSV path
                if data_source == "Sample Dataset":
                    input_csv = str(Path(__file__).parent.parent / "sample_data" / "marketing_campaign_performance_sample.csv")
                else:
                    # Save uploaded file temporarily
                    input_csv = "outputs/temp_upload.csv"
                    df.to_csv(input_csv, index=False)
                
                # Generate summary preview first (if Groq mode)
                summary = None
                latency = None
                ai_generated = False
                
                if use_groq and groq_api_key_set:
                    # Build context for Groq
                    from src.auto_report import build_llm_context
                    context = build_llm_context(df_filtered, kpis, top_campaigns, date_start, date_end)
                    
                    # Generate summary with Groq
                    from src.utils import llm_generate_summary_with_groq
                    summary, latency = llm_generate_summary_with_groq(context, use_cache=use_cache)
                    ai_generated = latency is not None
                else:
                    # Use template summary
                    from src.utils import _generate_template_summary
                    summary = _generate_template_summary(kpis, top_campaigns)
                
                # Generate report
                output_path = generate_report(
                    csv_path=input_csv,
                    output_path="outputs/automated_report.pptx",
                    date_start=date_start,
                    date_end=date_end,
                    campaigns=campaigns_list,
                    locations=locations_list,
                    channels=channels_list,
                    use_llm=False,  # Legacy mode disabled
                    use_groq=use_groq and groq_api_key_set,
                    use_cache=use_cache
                )
                
                st.success("✓ Report generated successfully!")
                
                # Show executive summary preview
                with st.expander("📝 Preview Executive Summary", expanded=True):
                    if ai_generated and latency:
                        st.success(f"🤖 AI summary generated by Groq (latency: {latency:.2f}s)")
                    st.text(summary)
                
                # Download buttons
                with open(output_path, "rb") as f:
                    pptx_bytes = f.read()
                
                # Try to convert to PDF (automatic - tries LibreOffice first, then Python libraries)
                pdf_path = output_path.replace('.pptx', '.pdf')
                pdf_generated = False
                pdf_bytes = None
                
                try:
                    from src.pdf_converter import convert_pptx_to_pdf
                    pdf_result = convert_pptx_to_pdf(output_path, pdf_path)
                    
                    if pdf_result and Path(pdf_path).exists():
                        pdf_generated = True
                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()
                        logger.info(f"PDF generated successfully: {pdf_path}")
                except Exception as e:
                    logger.warning(f"PDF conversion failed: {e}")
                
                # Display download buttons
                if pdf_generated and pdf_bytes:
                    # Both PPTX and PDF available
                    col_dl1, col_dl2 = st.columns(2)
                    
                    with col_dl1:
                        st.download_button(
                            label="📥 Download PPTX Report",
                            data=pptx_bytes,
                            file_name=f"automated_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pptx",
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            use_container_width=True
                        )
                    
                    with col_dl2:
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"automated_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                else:
                    # Only PPTX available
                    st.download_button(
                        label="📥 Download PPTX Report",
                        data=pptx_bytes,
                        file_name=f"automated_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        use_container_width=True
                    )
                    
                    # Show PDF conversion instructions (fallback)
                    with st.expander("📄 PDF Conversion Info", expanded=False):
                        st.info("💡 PDF conversion is attempted automatically using:")
                        st.markdown("""
                        1. **LibreOffice** (if installed) - Best quality
                        2. **Python libraries** (reportlab + Pillow) - Fallback method
                        
                        If PDF conversion failed, install the required libraries:
                        ```bash
                        pip install reportlab Pillow
                        ```
                        
                        Or install LibreOffice for best results:
                        - Download from: https://www.libreoffice.org/
                        """)
                
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
                logger.exception("Report generation failed")


if __name__ == "__main__":
    main()

