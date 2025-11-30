import tempfile
import pytest
from pathlib import Path
from spectral_physics.report import generate_markdown_report

def test_generate_markdown_report_basic():
    """Test basic report generation."""
    scores = {"ch1": 0.05, "ch2": 0.25}
    thresholds = {"ch1": 0.1, "ch2": 0.2}
    
    with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as f:
        temp_path = f.name
        
    try:
        generate_markdown_report(scores, thresholds, temp_path)
        
        with open(temp_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        assert "# Spectral Health Report" in content
        assert "| `ch1` | 0.050000 | 0.100000 | 🟢 OK |" in content
        assert "| `ch2` | 0.250000 | 0.200000 | 🔴 **ANOMALY** |" in content
        assert "Anomalies detected!" in content
        
    finally:
        Path(temp_path).unlink()

def test_generate_markdown_report_all_ok():
    """Test report generation when everything is OK."""
    scores = {"ch1": 0.05}
    thresholds = {"ch1": 0.1}
    
    with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as f:
        temp_path = f.name
        
    try:
        generate_markdown_report(scores, thresholds, temp_path)
        
        with open(temp_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        assert "All systems nominal" in content
        assert "Anomalies detected!" not in content
        
    finally:
        Path(temp_path).unlink()
