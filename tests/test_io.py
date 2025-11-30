import numpy as np
import pytest
import tempfile
from pathlib import Path
from spectral_physics.io import (
    load_timeseries_csv,
    save_spectrum_npz,
    load_spectrum_npz
)
from spectral_physics.spectrum import Spectrum1D


def test_load_timeseries_csv_simple():
    """Test loading simple CSV without header."""
    # Create temporary CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("1.0\n2.0\n3.0\n4.0\n5.0\n")
        temp_path = f.name
    
    try:
        signal = load_timeseries_csv(temp_path, column=0, skip_header=False)
        
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(signal, expected)
    finally:
        Path(temp_path).unlink()


def test_load_timeseries_csv_with_header():
    """Test loading CSV with header."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.css', delete=False) as f:
        f.write("time,value\n")
        f.write("0.0,1.0\n0.1,2.0\n0.2,3.0\n")
        temp_path = f.name
    
    try:
        signal = load_timeseries_csv(temp_path, column=1, skip_header=True)
        
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(signal, expected)
    finally:
        Path(temp_path).unlink()


def test_load_timeseries_csv_multiple_columns():
    """Test loading CSV with multiple columns."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("1.0,10.0,100.0\n2.0,20.0,200.0\n3.0,30.0,300.0\n")
        temp_path = f.name
    
    try:
        # Load column 1
        signal = load_timeseries_csv(temp_path, column=1, skip_header=False)
        expected = np.array([10.0, 20.0, 30.0])
        np.testing.assert_array_equal(signal, expected)
    finally:
        Path(temp_path).unlink()


def test_load_timeseries_csv_file_not_found():
    """Test that missing file raises ValueError."""
    with pytest.raises(ValueError, match="File not found"):
        load_timeseries_csv("nonexistent.csv")


def test_load_timeseries_csv_invalid_column():
    """Test that invalid column index raises ValueError."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("1.0,2.0\n3.0,4.0\n")
        temp_path = f.name
    
    try:
        with pytest.raises(ValueError, match="Column index .* out of range"):
            load_timeseries_csv(temp_path, column=5, skip_header=False)
    finally:
        Path(temp_path).unlink()


def test_save_load_spectrum_npz():
    """Test save and load spectrum roundtrip."""
    omega = np.array([1.0, 2.0, 3.0])
    power = np.array([0.5, 1.0, 0.5])
    
    original = Spectrum1D(omega=omega, power=power)
    
    # Save
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        temp_path = f.name
    
    try:
        save_spectrum_npz(original, temp_path)
        
        # Load
        loaded = load_spectrum_npz(temp_path)
        
        # Verify
        np.testing.assert_array_equal(loaded.omega, omega)
        np.testing.assert_array_equal(loaded.power, power)
    finally:
        Path(temp_path).unlink()


def test_load_spectrum_npz_missing_file():
    """Test that missing file raises ValueError."""
    with pytest.raises(ValueError, match="File not found"):
        load_spectrum_npz("nonexistent.npz")


def test_load_spectrum_npz_missing_keys():
    """Test that file missing required keys raises ValueError."""
    # Create npz with wrong keys
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        temp_path = f.name
    
    try:
        np.savez(temp_path, wrong_key=np.array([1, 2, 3]))
        
        with pytest.raises(ValueError, match="missing required keys"):
            load_spectrum_npz(temp_path)
    finally:
        Path(temp_path).unlink()
