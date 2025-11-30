"""
Material Spectrum Guess Demo

A "game" where the system tries to identify a material based on its LDOS signature.
1. We generate a synthetic LDOS map corresponding to a material (Steel, Water, Concrete).
2. We add noise to simulate real measurement.
3. The inference engine guesses the material.
"""

import numpy as np
import matplotlib.pyplot as plt
from spectral_physics.atoms import load_default_atom_db
from spectral_physics.materials import infer_material_from_ldos

def generate_mystery_ldos(material_type: str, nx=20, ny=20) -> np.ndarray:
    """
    Generate synthetic LDOS map for a given material type.
    """
    if material_type == "Steel":
        # Steel: Low mean (heavy), low variance (uniform)
        base = 0.04
        noise = 0.005
    elif material_type == "Water":
        # Water: High mean (light), high variance (liquid/disordered)
        base = 0.12
        noise = 0.03
    elif material_type == "Concrete":
        # Concrete: Medium mean, medium variance
        base = 0.08
        noise = 0.015
    else:
        raise ValueError(f"Unknown material: {material_type}")
        
    ldos = np.random.normal(base, noise, size=(ny, nx))
    return np.abs(ldos) # LDOS must be positive

def main():
    print("=" * 60)
    print("Material Spectrum Guess: AI Inference Demo")
    print("=" * 60)
    
    atom_db = load_default_atom_db()
    
    # Test cases
    test_materials = ["Steel", "Water", "Concrete"]
    
    for true_material in test_materials:
        print(f"\n--- Mystery Sample: {true_material} ---")
        
        # 1. Generate Data
        ldos_map = generate_mystery_ldos(true_material)
        print(f"Generated LDOS map. Mean: {np.mean(ldos_map):.4f}, Std: {np.std(ldos_map):.4f}")
        
        # 2. Infer
        candidates = infer_material_from_ldos(ldos_map, atom_db)
        
        # 3. Report
        print("AI Guesses:")
        top_guess = candidates[0]
        for i, cand in enumerate(candidates):
            print(f"  {i+1}. {cand.name:<20} Confidence: {cand.confidence:.2f}")
            
        # Check result
        # Note: Our inference names are "Steel (Fe-C)", "Water (H2O)", etc.
        if true_material in top_guess.name:
            print("✅ CORRECT IDENTIFICATION")
        else:
            print("❌ INCORRECT IDENTIFICATION")
            
    print("\n" + "=" * 60)
    print("Demo Complete.")

if __name__ == "__main__":
    main()
