"""
Utility functions for data processing, KPI calculations, charting, and LLM integration.
"""

import os
import json
import hashlib
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, will use system environment variables only
    pass

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


# ============================================================================
# Groq LLM Integration with Caching and Security
# ============================================================================

def sanitize_for_llm(aggregates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize aggregated data to ensure no PII is sent to LLM APIs.
    
    Removes any fields that could contain:
    - Customer identifiers (emails, phone numbers, IDs)
    - Raw user data
    - Personal information
    
    Args:
        aggregates: Dictionary of aggregated metrics
        
    Returns:
        Sanitized dictionary with only safe aggregated metrics
    """
    sanitized = {}
    
    # Only include safe aggregated metrics
    safe_keys = [
        'time_window', 'totals', 'metrics', 'top_campaigns',
        'total_impressions', 'total_clicks', 'total_spend', 'total_visits',
        'avg_ctr', 'avg_cpc', 'avg_roi'
    ]
    
    for key, value in aggregates.items():
        if key in safe_keys:
            # For top_campaigns, only include safe fields
            if key == 'top_campaigns' and isinstance(value, list):
                sanitized[key] = []
                for campaign in value:
                    safe_campaign = {
                        'name': str(campaign.get('Campaign', campaign.get('name', ''))),
                        'visits': float(campaign.get('Visits', campaign.get('visits', 0))),
                        'ctr': float(campaign.get('CTR', campaign.get('ctr', 0))),
                        'spend': float(campaign.get('Spend', campaign.get('spend', 0)))
                    }
                    sanitized[key].append(safe_campaign)
            else:
                sanitized[key] = value
    
    return sanitized


def _get_cache_path() -> Path:
    """Get path to LLM cache file."""
    cache_path = Path("outputs/llm_cache.json")
    cache_path.parent.mkdir(exist_ok=True)
    return cache_path


def _hash_context(context: Dict[str, Any]) -> str:
    """Generate SHA256 hash of context for caching."""
    context_str = json.dumps(context, sort_keys=True)
    return hashlib.sha256(context_str.encode()).hexdigest()


def _load_cache() -> Dict[str, Any]:
    """Load LLM cache from file."""
    cache_path = _get_cache_path()
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    """Save LLM cache to file."""
    cache_path = _get_cache_path()
    try:
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")


def clear_llm_cache() -> None:
    """Clear the LLM cache."""
    cache_path = _get_cache_path()
    if cache_path.exists():
        cache_path.unlink()
        logger.info("LLM cache cleared")
    else:
        logger.info("LLM cache already empty")


def _estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 characters)."""
    return len(text) // 4


def _truncate_context_if_needed(context: Dict[str, Any], max_tokens: int = 3000) -> Dict[str, Any]:
    """Truncate context if it exceeds token limit."""
    context_str = json.dumps(context)
    estimated_tokens = _estimate_tokens(context_str)
    
    if estimated_tokens <= max_tokens:
        return context
    
    logger.warning(f"Context too large ({estimated_tokens} tokens), truncating...")
    
    # Reduce top_campaigns to top 2 if present
    if 'top_campaigns' in context and isinstance(context['top_campaigns'], list):
        context['top_campaigns'] = context['top_campaigns'][:2]
    
    # Re-check
    context_str = json.dumps(context)
    estimated_tokens = _estimate_tokens(context_str)
    
    if estimated_tokens > max_tokens:
        # Further truncation: remove less critical fields
        if 'metrics' in context:
            context['metrics'] = {k: v for k, v in list(context['metrics'].items())[:3]}
    
    return context


def llm_generate_summary_with_groq(
    context: Dict[str, Any],
    use_cache: bool = True,
    timeout: int = 10
) -> Tuple[str, Optional[float]]:
    """
    Generate executive summary using Groq API with caching, retries, and fallback.
    
    Args:
        context: Dictionary with aggregated metrics (time_window, totals, metrics, top_campaigns)
        use_cache: Whether to use cached responses
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (summary_text, latency_seconds) or (template_summary, None) on failure
    """
    start_time = time.time()
    
    # Sanitize context to remove any PII
    sanitized_context = sanitize_for_llm(context)
    
    # Truncate if too large
    sanitized_context = _truncate_context_if_needed(sanitized_context)
    
    # Check cache
    if use_cache:
        cache = _load_cache()
        context_hash = _hash_context(sanitized_context)
        
        if context_hash in cache:
            cached_response = cache[context_hash]
            logger.info(f"Cache hit for context hash {context_hash[:8]}...")
            latency = time.time() - start_time
            cached_summary = cached_response.get('summary', '')
            return _normalize_summary_text(cached_summary), latency
    
    # Check for API key
    groq_key = os.getenv('GROQ_API_KEY')
    if not groq_key:
        logger.warning("GROQ_API_KEY not set, falling back to template")
        return generate_template_summary_from_context(sanitized_context), None
    
    # Build prompt
    prompt = _build_groq_prompt(sanitized_context)
    
    # Call Groq API with retries
    try:
        summary = _call_groq_with_retry(groq_key, prompt, timeout=timeout)
        
        # Validate response
        if not summary or len(summary.strip()) < 50:
            logger.warning("Groq returned empty or too short response, using template")
            return generate_template_summary_from_context(sanitized_context), None
        
        latency = time.time() - start_time
        
        # Cache the response
        if use_cache:
            cache = _load_cache()
            context_hash = _hash_context(sanitized_context)
            cache[context_hash] = {
                'summary': summary,
                'timestamp': datetime.now().isoformat(),
                'latency': latency
            }
            _save_cache(cache)
        
        logger.info(f"Groq summary generated successfully in {latency:.2f}s")
        return summary, latency
        
    except Exception as e:
        logger.error(f"Groq API call failed: {e}", exc_info=True)
        return generate_template_summary_from_context(sanitized_context), None


def _build_groq_prompt(context: Dict[str, Any]) -> str:
    """Build the prompt for Groq API."""
    prompt = """You are an executive summary writer for digital marketing performance. Produce a concise executive summary (3-4 sentences) and 3 bullet recommendations.

Context:
"""
    
    # Add time window
    if 'time_window' in context:
        prompt += f"Time Period: {context['time_window']}\n"
    
    # Add totals
    if 'totals' in context:
        totals = context['totals']
        prompt += f"\nAggregate Metrics:\n"
        prompt += f"- Total Impressions: {totals.get('impressions', 0):,.0f}\n"
        prompt += f"- Total Clicks: {totals.get('clicks', 0):,.0f}\n"
        prompt += f"- Total Spend: ${totals.get('spend', 0):,.2f}\n"
        prompt += f"- Total Visits: {totals.get('visits', 0):,.0f}\n"
    
    # Add metrics
    if 'metrics' in context:
        metrics = context['metrics']
        prompt += f"\nPerformance Metrics:\n"
        if 'avg_ctr' in metrics:
            prompt += f"- Average CTR: {metrics['avg_ctr']:.2f}%\n"
        if 'avg_cpc' in metrics:
            prompt += f"- Average CPC: ${metrics['avg_cpc']:.2f}\n"
        if 'avg_roi' in metrics:
            prompt += f"- Average ROI: {metrics['avg_roi']:.2f}\n"
    
    # Add top campaigns
    if 'top_campaigns' in context and context['top_campaigns']:
        prompt += f"\nTop Campaigns:\n"
        for i, campaign in enumerate(context['top_campaigns'][:3], 1):
            prompt += f"{i}. {campaign.get('name', 'N/A')}: {campaign.get('visits', 0):,.0f} visits, "
            prompt += f"{campaign.get('ctr', 0):.2f}% CTR, ${campaign.get('spend', 0):,.2f} spend\n"
    
    prompt += """
Constraints: Do not invent numbers; base your statements only on the given KPIs. Keep language formal and suitable for C-level executives.

Output format: Return output as JSON with keys: 'summary' (string), 'bullets' (array of strings), 'recommendations' (array of strings). If you cannot return JSON, return plain text with the summary, bullets, and recommendations clearly separated."""
    
    return prompt


def _call_groq_with_retry(api_key: str, prompt: str, timeout: int = 10, max_retries: int = 2) -> str:
    """Call Groq API with exponential backoff retries."""
    try:
        from groq import Groq
    except ImportError:
        raise Exception("groq package not installed. Install with: pip install groq")
    
    client = Groq(api_key=api_key, timeout=timeout)
    
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Calling Groq API (attempt {attempt + 1}/{max_retries + 1})")
            
            # Use a current, supported Groq model
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )
            
            raw_text = response.choices[0].message.content.strip()
            formatted_output = _format_llm_output(raw_text)
            
            return formatted_output
            
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                logger.warning(f"Groq API call failed (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Groq API call failed after {max_retries + 1} attempts: {e}")
    
    raise last_exception


def _normalize_summary_text(text: str) -> str:
    """Ensure cached summaries are rendered without JSON fences."""
    if not text:
        return text
    return _format_llm_output(text)


def _extract_json_from_text(raw_text: str) -> Optional[dict]:
    """Extract JSON payload from LLM output."""
    cleaned = raw_text.strip()
    
    if "```json" in cleaned:
        start = cleaned.find("```json") + len("```json")
        end = cleaned.find("```", start)
        if end != -1:
            candidate = cleaned[start:end].strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    
    stripped = cleaned
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _format_llm_output(raw_text: str) -> str:
    """Format Groq output into plain text without JSON fences."""
    payload = _extract_json_from_text(raw_text)
    if payload:
        parts = []
        summary = payload.get("summary")
        if summary:
            parts.append(summary.strip())
        bullets = payload.get("bullets")
        if bullets:
            parts.append("\nKey Insights:")
            for bullet in bullets:
                parts.append(f"• {bullet}")
        recs = payload.get("recommendations")
        if recs:
            parts.append("\nRecommendations:")
            for rec in recs:
                parts.append(f"• {rec}")
        formatted = "\n".join(parts).strip()
        if formatted:
            return formatted
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return raw_text


def _extract_json_from_text(raw_text: str) -> Optional[dict]:
    """Extract JSON payload from LLM output."""
    cleaned = raw_text.strip()
    
    if "```json" in cleaned:
        start = cleaned.find("```json") + len("```json")
        end = cleaned.find("```", start)
        if end != -1:
            candidate = cleaned[start:end].strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    
    # Try entire text (removing fences if needed)
    stripped = cleaned
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _format_llm_output(raw_text: str) -> str:
    """Format Groq output into plain text without JSON fences."""
    payload = _extract_json_from_text(raw_text)
    if payload:
        parts = []
        summary = payload.get("summary")
        if summary:
            parts.append(summary.strip())
        bullets = payload.get("bullets")
        if bullets:
            parts.append("\nKey Insights:")
            for bullet in bullets:
                parts.append(f"• {bullet}")
        recs = payload.get("recommendations")
        if recs:
            parts.append("\nRecommendations:")
            for rec in recs:
                parts.append(f"• {rec}")
        formatted = "\n".join(parts).strip()
        if formatted:
            return formatted
    # No JSON payload detected; remove fences if present
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return raw_text


def generate_template_summary_from_context(context: Dict[str, Any]) -> str:
    """Generate template summary from context dictionary."""
    totals = context.get('totals', {})
    metrics = context.get('metrics', {})
    top_campaigns = context.get('top_campaigns', [])
    
    summary = "EXECUTIVE SUMMARY\n\n"
    summary += "Key Insights:\n"
    
    visits = totals.get('visits', 0)
    ctr = metrics.get('avg_ctr', 0)
    spend = totals.get('spend', 0)
    clicks = totals.get('clicks', 0)
    cpc = metrics.get('avg_cpc', 0)
    
    summary += f"• Campaign performance shows {visits:,.0f} total visits with an average CTR of {ctr:.2f}%\n"
    summary += f"• Total marketing spend of ${spend:,.2f} generated {clicks:,.0f} clicks at an average CPC of ${cpc:.2f}\n"
    
    if top_campaigns:
        top = top_campaigns[0]
        summary += f"• Top performing campaign: {top.get('name', 'N/A')} with {top.get('visits', 0):,.0f} visits and {top.get('ctr', 0):.2f}% CTR\n"
    
    summary += "\nRecommendations:\n"
    if ctr < 2.0:
        summary += "• Optimize ad creative and targeting to improve click-through rates\n"
    else:
        summary += "• Maintain current creative performance and scale successful campaigns\n"
    
    if cpc > 1.0:
        summary += "• Review bidding strategy and keyword selection to reduce cost per click\n"
    else:
        summary += "• Current CPC is efficient; consider increasing budget for high-performing campaigns\n"
    
    return summary

