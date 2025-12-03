"""
Tests for LLM integration with Groq.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    llm_generate_summary_with_groq,
    sanitize_for_llm,
    clear_llm_cache,
    _hash_context
)


def test_sanitize_for_llm():
    """Test that sanitize_for_llm removes PII and keeps only safe aggregated metrics."""
    aggregates = {
        'time_window': '2024-01-01 to 2024-01-30',
        'totals': {
            'impressions': 1000000,
            'clicks': 50000,
            'spend': 10000.50,
            'visits': 45000
        },
        'metrics': {
            'avg_ctr': 5.0,
            'avg_cpc': 0.20,
            'avg_roi': 4.5
        },
        'top_campaigns': [
            {
                'Campaign': 'Test Campaign',
                'Visits': 1000,
                'CTR': 3.5,
                'Spend': 500.0
            }
        ],
        'unsafe_field': 'should_be_removed',
        'customer_email': 'test@example.com'  # Should be removed
    }
    
    sanitized = sanitize_for_llm(aggregates)
    
    # Check that unsafe fields are removed
    assert 'unsafe_field' not in sanitized
    assert 'customer_email' not in sanitized
    
    # Check that safe fields are present
    assert 'time_window' in sanitized
    assert 'totals' in sanitized
    assert 'metrics' in sanitized
    assert 'top_campaigns' in sanitized
    
    # Check top_campaigns structure
    assert len(sanitized['top_campaigns']) == 1
    campaign = sanitized['top_campaigns'][0]
    assert 'name' in campaign
    assert 'visits' in campaign
    assert 'ctr' in campaign
    assert 'spend' in campaign


def test_hash_context():
    """Test that context hashing produces consistent results."""
    context1 = {'time_window': '2024-01-01 to 2024-01-30', 'totals': {'impressions': 1000}}
    context2 = {'time_window': '2024-01-01 to 2024-01-30', 'totals': {'impressions': 1000}}
    context3 = {'time_window': '2024-01-01 to 2024-01-30', 'totals': {'impressions': 2000}}
    
    hash1 = _hash_context(context1)
    hash2 = _hash_context(context2)
    hash3 = _hash_context(context3)
    
    # Same context should produce same hash
    assert hash1 == hash2
    
    # Different context should produce different hash
    assert hash1 != hash3


@patch('src.utils.Groq')
def test_llm_generate_summary_with_groq_success(mock_groq_class):
    """Test successful Groq API call with JSON response."""
    # Mock Groq client
    mock_client = MagicMock()
    mock_groq_class.return_value = mock_client
    
    # Mock response with JSON
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        'summary': 'Test summary text',
        'bullets': ['Insight 1', 'Insight 2', 'Insight 3'],
        'recommendations': ['Rec 1', 'Rec 2']
    })
    mock_client.chat.completions.create.return_value = mock_response
    
    # Set environment variable
    import os
    with patch.dict(os.environ, {'GROQ_API_KEY': 'test_key'}):
        context = {
            'time_window': '2024-01-01 to 2024-01-30',
            'totals': {'impressions': 1000, 'clicks': 50, 'spend': 100, 'visits': 45},
            'metrics': {'avg_ctr': 5.0, 'avg_cpc': 2.0},
            'top_campaigns': []
        }
        
        summary, latency = llm_generate_summary_with_groq(context, use_cache=False)
        
        # Check that summary contains expected content
        assert 'Test summary text' in summary
        assert 'Insight 1' in summary or 'Insight 2' in summary
        assert latency is not None
        assert latency > 0


@patch('src.utils.Groq')
def test_llm_generate_summary_with_groq_plain_text(mock_groq_class):
    """Test Groq API call with plain text response (not JSON)."""
    # Mock Groq client
    mock_client = MagicMock()
    mock_groq_class.return_value = mock_client
    
    # Mock response with plain text
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "This is a plain text summary with insights."
    mock_client.chat.completions.create.return_value = mock_response
    
    # Set environment variable
    import os
    with patch.dict(os.environ, {'GROQ_API_KEY': 'test_key'}):
        context = {
            'time_window': '2024-01-01 to 2024-01-30',
            'totals': {'impressions': 1000, 'clicks': 50, 'spend': 100, 'visits': 45},
            'metrics': {'avg_ctr': 5.0, 'avg_cpc': 2.0},
            'top_campaigns': []
        }
        
        summary, latency = llm_generate_summary_with_groq(context, use_cache=False)
        
        # Check that plain text is returned
        assert 'plain text summary' in summary.lower()
        assert latency is not None


@patch('src.utils.Groq')
def test_llm_generate_summary_with_groq_fallback_on_error(mock_groq_class):
    """Test that errors fall back to template summary."""
    # Mock Groq client to raise exception
    mock_groq_class.side_effect = Exception("API Error")
    
    # Set environment variable
    import os
    with patch.dict(os.environ, {'GROQ_API_KEY': 'test_key'}):
        context = {
            'time_window': '2024-01-01 to 2024-01-30',
            'totals': {'impressions': 1000, 'clicks': 50, 'spend': 100, 'visits': 45},
            'metrics': {'avg_ctr': 5.0, 'avg_cpc': 2.0},
            'top_campaigns': []
        }
        
        summary, latency = llm_generate_summary_with_groq(context, use_cache=False)
        
        # Should return template summary (not None)
        assert summary is not None
        assert len(summary) > 0
        assert 'EXECUTIVE SUMMARY' in summary or 'Key Insights' in summary
        # Latency should be None on fallback
        assert latency is None


def test_llm_generate_summary_with_groq_no_api_key():
    """Test that missing API key falls back to template."""
    import os
    # Ensure no API key is set
    with patch.dict(os.environ, {}, clear=True):
        context = {
            'time_window': '2024-01-01 to 2024-01-30',
            'totals': {'impressions': 1000, 'clicks': 50, 'spend': 100, 'visits': 45},
            'metrics': {'avg_ctr': 5.0, 'avg_cpc': 2.0},
            'top_campaigns': []
        }
        
        summary, latency = llm_generate_summary_with_groq(context, use_cache=False)
        
        # Should return template summary
        assert summary is not None
        assert len(summary) > 0
        assert latency is None


@patch('src.utils.Groq')
def test_llm_generate_summary_with_groq_empty_response_fallback(mock_groq_class):
    """Test that empty or too short responses fall back to template."""
    # Mock Groq client
    mock_client = MagicMock()
    mock_groq_class.return_value = mock_client
    
    # Mock response with empty content
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = ""  # Empty response
    mock_client.chat.completions.create.return_value = mock_response
    
    # Set environment variable
    import os
    with patch.dict(os.environ, {'GROQ_API_KEY': 'test_key'}):
        context = {
            'time_window': '2024-01-01 to 2024-01-30',
            'totals': {'impressions': 1000, 'clicks': 50, 'spend': 100, 'visits': 45},
            'metrics': {'avg_ctr': 5.0, 'avg_cpc': 2.0},
            'top_campaigns': []
        }
        
        summary, latency = llm_generate_summary_with_groq(context, use_cache=False)
        
        # Should fall back to template
        assert summary is not None
        assert len(summary) > 50  # Template should be substantial
        assert latency is None


def test_clear_llm_cache():
    """Test clearing the LLM cache."""
    # This should not raise an error even if cache doesn't exist
    clear_llm_cache()
    
    # Call again (should still work)
    clear_llm_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

