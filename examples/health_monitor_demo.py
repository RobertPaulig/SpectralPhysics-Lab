"""
Health Monitor Demo (Pump Monitoring)

This script demonstrates the end-to-end spectral-health pipeline:
1. Generate synthetic pump vibration data
2. Train a health profile
3. Score current data
4. Display the generated report
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("Pump Health Monitoring Demo")
    print("=" * 60)
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # 1. Generate synthetic data
    print("\n[Step 1/3] Generating synthetic pump data...")
    result = subprocess.run(
        [sys.executable, str(project_root / "examples" / "generate_synthetic_pump_data.py")],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("Error:", result.stderr)
        return
    print("✓ Data generated")
    
    # 2. Train health profile
    print("\n[Step 2/3] Training health profile...")
    result = subprocess.run(
        [
            "spectral-health", "train",
            "--config", str(project_root / "configs" / "pump_train.yaml"),
            "--out", str(project_root / "data" / "pump" / "profile.npz")
        ],
        capture_output=True,
        text=True,
        cwd=str(project_root)
    )
    if result.returncode != 0:
        print("Error:", result.stderr)
        return
    print("✓ Profile trained")
    
    # 3. Score current data
    print("\n[Step 3/3] Scoring current data and generating report...")
    result = subprocess.run(
        [
            "spectral-health", "score",
            "--config", str(project_root / "configs" / "pump_score.yaml"),
            "--profile", str(project_root / "data" / "pump" / "profile.npz"),
            "--thresholds", str(project_root / "configs" / "pump_thresholds.yaml"),
            "--report", str(project_root / "data" / "pump" / "report.md")
        ],
        capture_output=True,
        text=True,
        cwd=str(project_root)
    )
    if result.returncode != 0:
        print("Error:", result.stderr)
        return
    print("✓ Report generated")
    
    # 4. Display report
    print("\n" + "=" * 60)
    print("HEALTH REPORT:")
    print("=" * 60)
    report_path = project_root / "data" / "pump" / "report.md"
    if report_path.exists():
        print(report_path.read_text())
    else:
        print("Report not found!")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
