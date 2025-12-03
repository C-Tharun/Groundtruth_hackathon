"""
Utility functions for data processing, KPI calculations, charting, and LLM integration.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_csv_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that the CSV contains required columns for marketing campaign analysis.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_columns = ['Date', 'Campaign', 'Impressions', 'Clicks', 'Spend', 'Visits']
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    
    # Check for empty dataframe
    if df.empty:
        return False, "DataFrame is empty"
    
    # Check for required numeric columns
    numeric_cols = ['Impressions', 'Clicks', 'Spend', 'Visits']
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                return False, f"Column {col} cannot be converted to numeric: {str(e)}"
    
    return True, ""


def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute key performance indicators from marketing campaign data.
    
    Args:
        df: DataFrame with marketing campaign data
        
    Returns:
        Dictionary of KPI metrics
    """
    logger.info("Computing KPIs from data")
    
    # Ensure numeric columns
    numeric_cols = ['Impressions', 'Clicks', 'Spend', 'Visits']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate KPIs
    total_impressions = float(df['Impressions'].sum())
    total_clicks = float(df['Clicks'].sum())
    total_spend = float(df['Spend'].sum())
    total_visits = float(df['Visits'].sum())
    
    # Average CTR (Click-Through Rate)
    avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0.0
    
    # Average CPC (Cost Per Click)
    avg_cpc = (total_spend / total_clicks) if total_clicks > 0 else 0.0
    
    kpis = {
        'total_impressions': total_impressions,
        'total_clicks': total_clicks,
        'total_spend': total_spend,
        'total_visits': total_visits,
        'avg_ctr': avg_ctr,
        'avg_cpc': avg_cpc
    }
    
    logger.info(f"Computed KPIs: {kpis}")
    return kpis


def create_time_series_chart(df: pd.DataFrame, output_path: str) -> str:
    """
    Create a time series chart showing visits and impressions over time.
    
    Args:
        df: DataFrame with Date, Visits, and Impressions columns
        output_path: Path to save the chart image
        
    Returns:
        Path to saved chart
    """
    logger.info("Creating time series chart")
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df_sorted = df.sort_values('Date')
    
    # Aggregate by date
    daily = df_sorted.groupby('Date').agg({
        'Visits': 'sum',
        'Impressions': 'sum'
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(daily['Date'], daily['Visits'], label='Visits', marker='o', linewidth=2, color='#2E86AB')
    ax.plot(daily['Date'], daily['Impressions'], label='Impressions', marker='s', linewidth=2, color='#A23B72', alpha=0.7)
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Marketing Performance Over Time', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(daily) // 10)))
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Time series chart saved to {output_path}")
    return output_path


def create_top_campaigns_chart(df: pd.DataFrame, output_path: str, top_n: int = 5) -> str:
    """
    Create a bar chart showing top N campaigns by visits.
    
    Args:
        df: DataFrame with Campaign and Visits columns
        output_path: Path to save the chart image
        top_n: Number of top campaigns to show
        
    Returns:
        Path to saved chart
    """
    logger.info(f"Creating top {top_n} campaigns chart")
    
    # Aggregate by campaign
    campaign_stats = df.groupby('Campaign').agg({
        'Visits': 'sum'
    }).reset_index()
    
    # Sort and get top N
    top_campaigns = campaign_stats.nlargest(top_n, 'Visits')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(range(len(top_campaigns)), top_campaigns['Visits'], color='#06A77D')
    ax.set_yticks(range(len(top_campaigns)))
    ax.set_yticklabels(top_campaigns['Campaign'], fontsize=10)
    ax.set_xlabel('Total Visits', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Campaigns by Visits', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_campaigns.iterrows()):
        ax.text(row['Visits'], i, f"{int(row['Visits']):,}", 
                va='center', ha='left', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Top campaigns chart saved to {output_path}")
    return output_path


def get_top_campaigns_list(df: pd.DataFrame, top_n: int = 5) -> List[Dict[str, any]]:
    """
    Get top N campaigns with their key metrics.
    
    Args:
        df: DataFrame with campaign data
        top_n: Number of top campaigns to return
        
    Returns:
        List of dictionaries with campaign metrics
    """
    campaign_stats = df.groupby('Campaign').agg({
        'Visits': 'sum',
        'Clicks': 'sum',
        'Spend': 'sum',
        'Impressions': 'sum'
    }).reset_index()
    
    campaign_stats['CTR'] = (campaign_stats['Clicks'] / campaign_stats['Impressions'] * 100).fillna(0)
    campaign_stats = campaign_stats.sort_values('Visits', ascending=False).head(top_n)
    
    return campaign_stats.to_dict('records')


def llm_generate_summary(kpis: Dict[str, float], top_campaigns: List[Dict[str, any]], 
                        use_llm: bool = True) -> str:
    """
    Generate an executive summary using LLM (if available) or fallback template.
    
    Args:
        kpis: Dictionary of computed KPIs
        top_campaigns: List of top campaign dictionaries
        use_llm: Whether to attempt LLM generation (default: True)
        
    Returns:
        Executive summary text
    """
    logger.info(f"Generating executive summary (LLM mode: {use_llm})")
    
    if not use_llm:
        return _generate_template_summary(kpis, top_campaigns)
    
    # Try Groq first (free tier available)
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key:
        try:
            return _call_groq_api(kpis, top_campaigns, groq_key)
        except Exception as e:
            logger.warning(f"Groq API call failed: {e}, falling back to template")
    
    # Try OpenAI as fallback
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        try:
            return _call_openai_api(kpis, top_campaigns, openai_key)
        except Exception as e:
            logger.warning(f"OpenAI API call failed: {e}, falling back to template")
    
    # Fallback to template
    logger.info("No API keys found or API calls failed, using template summary")
    return _generate_template_summary(kpis, top_campaigns)


def _call_groq_api(kpis: Dict[str, float], top_campaigns: List[Dict[str, any]], 
                   api_key: str) -> str:
    """Call Groq API to generate summary."""
    try:
        from groq import Groq
        
        client = Groq(api_key=api_key)
        
        # Prepare aggregated metrics (no PII)
        prompt = f"""Generate a concise executive summary for a marketing campaign performance report.

Key Metrics:
- Total Impressions: {kpis['total_impressions']:,.0f}
- Total Clicks: {kpis['total_clicks']:,.0f}
- Total Spend: ${kpis['total_spend']:,.2f}
- Total Visits: {kpis['total_visits']:,.0f}
- Average CTR: {kpis['avg_ctr']:.2f}%
- Average CPC: ${kpis['avg_cpc']:.2f}

Top Campaign: {top_campaigns[0]['Campaign'] if top_campaigns else 'N/A'} with {top_campaigns[0]['Visits']:,.0f} visits

Provide:
1. Three key insights (bullet points)
2. Two actionable recommendations

Format as plain text, no markdown."""
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        
        summary = response.choices[0].message.content.strip()
        logger.info("Successfully generated summary using Groq API")
        return summary
        
    except ImportError:
        raise Exception("groq package not installed")
    except Exception as e:
        raise Exception(f"Groq API error: {str(e)}")


def _call_openai_api(kpis: Dict[str, float], top_campaigns: List[Dict[str, any]], 
                     api_key: str) -> str:
    """Call OpenAI API to generate summary."""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        
        prompt = f"""Generate a concise executive summary for a marketing campaign performance report.

Key Metrics:
- Total Impressions: {kpis['total_impressions']:,.0f}
- Total Clicks: {kpis['total_clicks']:,.0f}
- Total Spend: ${kpis['total_spend']:,.2f}
- Total Visits: {kpis['total_visits']:,.0f}
- Average CTR: {kpis['avg_ctr']:.2f}%
- Average CPC: ${kpis['avg_cpc']:.2f}

Top Campaign: {top_campaigns[0]['Campaign'] if top_campaigns else 'N/A'} with {top_campaigns[0]['Visits']:,.0f} visits

Provide:
1. Three key insights (bullet points)
2. Two actionable recommendations

Format as plain text, no markdown."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        
        summary = response.choices[0].message.content.strip()
        logger.info("Successfully generated summary using OpenAI API")
        return summary
        
    except ImportError:
        raise Exception("openai package not installed")
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")


def _generate_template_summary(kpis: Dict[str, float], top_campaigns: List[Dict[str, any]]) -> str:
    """Generate a deterministic template-based summary."""
    top_campaign = top_campaigns[0] if top_campaigns else None
    
    summary = "EXECUTIVE SUMMARY\n\n"
    summary += "Key Insights:\n"
    summary += f"• Campaign performance shows {kpis['total_visits']:,.0f} total visits with an average CTR of {kpis['avg_ctr']:.2f}%\n"
    summary += f"• Total marketing spend of ${kpis['total_spend']:,.2f} generated {kpis['total_clicks']:,.0f} clicks at an average CPC of ${kpis['avg_cpc']:.2f}\n"
    if top_campaign:
        summary += f"• Top performing campaign: {top_campaign['Campaign']} with {top_campaign['Visits']:,.0f} visits and {top_campaign['CTR']:.2f}% CTR\n"
    
    summary += "\nRecommendations:\n"
    if kpis['avg_ctr'] < 2.0:
        summary += "• Optimize ad creative and targeting to improve click-through rates\n"
    else:
        summary += "• Maintain current creative performance and scale successful campaigns\n"
    
    if kpis['avg_cpc'] > 1.0:
        summary += "• Review bidding strategy and keyword selection to reduce cost per click\n"
    else:
        summary += "• Current CPC is efficient; consider increasing budget for high-performing campaigns\n"
    
    return summary

