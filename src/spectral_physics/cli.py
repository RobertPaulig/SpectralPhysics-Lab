import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any

from .io import load_timeseries_csv, save_health_profile, load_health_profile
from .diagnostics import (
    ChannelConfig, 
    SpectralAnalyzer, 
    build_health_profile,
)
from .material import HealthProfile

from .spectrum import Spectrum1D


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train_command(args):
    """Execute training command."""
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    dt = config.get('dt')
    window = config.get('window', 'hann')
    channels_conf = config.get('channels', {})
    
    if not channels_conf:
        print("Error: No channels defined in config.")
        return 1
        
    training_data: Dict[str, List[Spectrum1D]] = {}
    
    print(f"Processing {len(channels_conf)} channels...")
    
    for name, ch_data in channels_conf.items():
        print(f"  Channel '{name}':")
        column = ch_data.get('column', 0)
        files = ch_data.get('files', [])
        
        # Create analyzer for this channel
        ch_config = ChannelConfig(
            name=name,
            dt=dt,
            window=window,
            freq_min=ch_data.get('freq_min'),
            freq_max=ch_data.get('freq_max')
        )
        analyzer = SpectralAnalyzer(ch_config)
        
        spectra = []
        for file_path in files:
            try:
                print(f"    Loading {file_path}...", end='', flush=True)
                signal = load_timeseries_csv(file_path, column=column)
                spec = analyzer.analyze(signal)
                spectra.append(spec)
                print(" OK")
            except Exception as e:
                print(f" FAIL: {e}")
        
        if spectra:
            training_data[name] = spectra
        else:
            print(f"    Warning: No valid data for channel '{name}'")

    if not training_data:
        print("Error: No training data collected.")
        return 1
        
    print("Building health profile...")
    profile = build_health_profile(training_data)
    
    print(f"Saving profile to {args.out}...")
    save_health_profile(profile, args.out)
    print("Done.")
    return 0


def score_command(args):
    """Execute scoring command."""
    print(f"Loading profile from {args.profile}...")
    try:
        profile = load_health_profile(args.profile)
    except Exception as e:
        print(f"Error loading profile: {e}")
        return 2

    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    print(f"Loading thresholds from {args.thresholds}...")
    thresholds = load_config(args.thresholds)
    
    dt = config.get('dt')
    window = config.get('window', 'hann')
    channels_conf = config.get('channels', {})
    
    current_spectra: Dict[str, Spectrum1D] = {}
    
    # Process current data (take first file from list or 'current' key)
    for name, ch_data in channels_conf.items():
        if name not in profile.signatures:
            continue
            
        files = ch_data.get('files', [])
        if not files:
            continue
            
        # Use the last file as "current" state
        file_path = files[-1]
        column = ch_data.get('column', 0)
        
        ch_config = ChannelConfig(
            name=name,
            dt=dt,
            window=window,
            freq_min=ch_data.get('freq_min'),
            freq_max=ch_data.get('freq_max')
        )
        analyzer = SpectralAnalyzer(ch_config)
        
        try:
            signal = load_timeseries_csv(file_path, column=column)
            spec = analyzer.analyze(signal)
            current_spectra[name] = spec
        except Exception as e:
            print(f"Error processing channel {name}: {e}")

    if not current_spectra:
        print("Error: No current data to analyze.")
        return 1

    # Calculate scores
    scores = profile.score(current_spectra)
    anomalies = profile.is_anomalous(current_spectra, thresholds)
    
    # Generate report
    print("\n" + "=" * 60)
    print(f"{'Channel':<15} {'Distance':<12} {'Threshold':<12} {'Status':<10}")
    print("-" * 60)
    
    any_anomaly = False
    
    for name in scores:
        dist = scores[name]
        thresh = thresholds.get(name, 0.0)
        is_anom = anomalies.get(name, False)
        status = "ANOMALY" if is_anom else "OK"
        
        if is_anom:
            any_anomaly = True
            
        print(f"{name:<15} {dist:.6f}     {thresh:.6f}     {status:<10}")
        
    print("=" * 60)
    
    if args.report:
        from .report import generate_markdown_report
        print(f"Generating report: {args.report}")
        generate_markdown_report(scores, thresholds, args.report)

    return 1 if any_anomaly else 0


def main():
    parser = argparse.ArgumentParser(
        description="Spectral Health: Multi-channel diagnostics engine"
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Train command
    parser_train = subparsers.add_parser('train', help='Train health profile')
    parser_train.add_argument('--config', required=True, help='Path to config.yaml')
    parser_train.add_argument('--out', required=True, help='Output path for profile.npz')
    parser_train.set_defaults(func=train_command)
    
    # Score command
    parser_score = subparsers.add_parser('score', help='Score current data against profile')
    parser_score.add_argument('--config', required=True, help='Path to config.yaml')
    parser_score.add_argument('--profile', required=True, help='Path to profile.npz')
    parser_score.add_argument('--thresholds', required=True, help='Path to thresholds.yaml')
    parser_score.add_argument('--report', help='Path to output markdown report')
    parser_score.set_defaults(func=score_command)
    
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
