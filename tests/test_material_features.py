import numpy as np
import pytest
from spectral_physics.spectrum import Spectrum1D
from spectral_physics.material import FeatureSignature, HealthProfile, MaterialSignature

def test_feature_signature_distance():
    ref_feats = np.array([1.0, 2.0, 3.0])
    sig = FeatureSignature(reference_features=ref_feats)
    
    # Identical
    assert sig.distance_l2(ref_feats) == 0.0
    
    # Different
    other = np.array([2.0, 2.0, 3.0]) # diff 1.0 in first component
    assert sig.distance_l2(other) == 1.0
    
    # Shape mismatch
    with pytest.raises(ValueError):
        sig.distance_l2(np.array([1.0, 2.0]))

def test_health_profile_score_features(monkeypatch):
    # Mock extract_features to return predictable values
    import spectral_physics.material as mat
    
    # We need to mock it where it is imported inside the method or globally?
    # The method does: from .diagnostics import extract_features
    # So we need to mock spectral_physics.diagnostics.extract_features
    
    # But since it's a local import inside the function, mocking might be tricky if we don't patch the module.
    # Let's just use real objects, it's safer.
    
    # Setup
    omega = np.linspace(0, 10, 11)
    power = np.ones_like(omega)
    spec = Spectrum1D(omega, power)
    
    # Create profile with feature signature
    # Assume extract_features returns [band1, entropy] -> length 2 for 1 band
    # For power=1 everywhere, band power depends on width.
    # Let's just create a dummy FeatureSignature that matches what we expect from "healthy"
    
    # Let's compute expected features for "healthy"
    # Band 0-10 (all points? freq_hz = omega/2pi)
    # omega=10 -> freq ~ 1.59 Hz.
    # Band 0-2 Hz covers everything.
    # Power sum = 11.
    # Entropy = ln(11).
    
    bands = {'ch1': [(0.0, 2.0)]}
    
    # "Healthy" features
    feat_ref = np.array([11.0, np.log(11)])
    
    feat_sig = FeatureSignature(reference_features=feat_ref)
    
    profile = HealthProfile(
        signatures={'ch1': MaterialSignature(reference=spec)},
        feature_signatures={'ch1': feat_sig}
    )
    
    # 1. Test with identical spectrum
    scores = profile.score_features({'ch1': spec}, bands)
    assert 'ch1' in scores
    assert np.isclose(scores['ch1'], 0.0)
    
    # 2. Test with modified spectrum
    # Double the power -> band power doubles (22), entropy stays same (normalized)
    spec2 = Spectrum1D(omega, power * 2)
    scores2 = profile.score_features({'ch1': spec2}, bands)
    
    # Diff: band power 22 vs 11 -> diff 11. Entropy same.
    # Distance = sqrt(11^2 + 0) = 11.
    assert np.isclose(scores2['ch1'], 11.0)
