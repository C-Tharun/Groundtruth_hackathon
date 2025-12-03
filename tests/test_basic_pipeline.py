"""
Basic smoke tests for the automated report pipeline.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import validate_csv_data, compute_kpis


def test_load_sample_data():
    """Test that sample data can be loaded."""
    sample_path = Path(__file__).parent.parent / "sample_data" / "marketing_campaign_performance_sample.csv"
    assert sample_path.exists(), "Sample data file not found"
    
    df = pd.read_csv(sample_path)
    assert not df.empty, "Sample data is empty"
    assert len(df) > 0, "Sample data has no rows"


def test_validate_csv_data():
    """Test CSV data validation."""
    sample_path = Path(__file__).parent.parent / "sample_data" / "marketing_campaign_performance_sample.csv"
    df = pd.read_csv(sample_path)
    
    is_valid, error_msg = validate_csv_data(df)
    assert is_valid, f"Data validation failed: {error_msg}"


def test_compute_kpis():
    """Test KPI computation on sample data."""
    sample_path = Path(__file__).parent.parent / "sample_data" / "marketing_campaign_performance_sample.csv"
    df = pd.read_csv(sample_path)
    
    kpis = compute_kpis(df)
    
    # Assert all expected KPI keys exist
    expected_keys = [
        'total_impressions',
        'total_clicks',
        'total_spend',
        'total_visits',
        'avg_ctr',
        'avg_cpc'
    ]
    
    for key in expected_keys:
        assert key in kpis, f"Missing KPI: {key}"
        assert isinstance(kpis[key], (int, float)), f"KPI {key} is not numeric"
    
    # Assert values are reasonable
    assert kpis['total_impressions'] > 0, "Total impressions should be positive"
    assert kpis['total_clicks'] > 0, "Total clicks should be positive"
    assert kpis['total_spend'] > 0, "Total spend should be positive"
    assert kpis['total_visits'] > 0, "Total visits should be positive"
    assert kpis['avg_ctr'] >= 0, "Average CTR should be non-negative"
    assert kpis['avg_cpc'] >= 0, "Average CPC should be non-negative"


def test_required_columns():
    """Test that required columns are present."""
    sample_path = Path(__file__).parent.parent / "sample_data" / "marketing_campaign_performance_sample.csv"
    df = pd.read_csv(sample_path)
    
    required_columns = ['Date', 'Campaign', 'Impressions', 'Clicks', 'Spend', 'Visits']
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

