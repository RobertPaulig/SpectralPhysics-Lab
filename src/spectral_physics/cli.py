import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any

from .io import (
    load_timeseries_csv, 
    save_health_profile, 
    load_health_profile,
    save_ndt_profile,
    load_ndt_profile
)
from .diagnostics import (
    ChannelConfig, 
    SpectralAnalyzer, 
    build_health_profile,
)
from .material import HealthProfile
from .ndt import build_ndt_profile, score_ndt_state, ndt_defect_mask
from .medium_2d import OscillatorGrid2D
from .geophysics_2d import GeoGrid2D

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


def ndt_train_command(args):
    """Execute NDT training command."""
    print(f"Loading grid config from {args.grid_config}...")
    config = load_config(args.grid_config)
    
    # Extract grid parameters
    nx = config.get('nx', 20)
    ny = config.get('ny', 20)
    kx = config.get('kx', 1.0)
    ky = config.get('ky', 1.0)
    m = config.get('m', 1.0)
    
    print(f"Building OscillatorGrid2D ({nx}x{ny})...")
    grid = OscillatorGrid2D(nx=nx, ny=ny, kx=kx, ky=ky, m=m)
    
    # NDT parameters
    n_modes = config.get('n_modes', 50)
    freq_window = tuple(config.get('freq_window', [0.0, 1.5]))
    n_samples = config.get('n_samples', 1)
    noise_level = config.get('noise_level', 0.0)
    
    print(f"Building NDT profile (modes={n_modes}, window={freq_window})...")
    profile = build_ndt_profile(
        grid=grid,
        n_modes=n_modes,
        freq_window=freq_window,
        n_samples=n_samples,
        noise_level=noise_level
    )
    
    print(f"Saving profile to {args.profile_out}...")
    save_ndt_profile(profile, args.profile_out)
    print("Done.")
    return 0


def ndt_score_command(args):
    """Execute NDT scoring command."""
    print(f"Loading profile from {args.profile}...")
    try:
        profile = load_ndt_profile(args.profile)
    except Exception as e:
        print(f"Error loading profile: {e}")
        return 2
        
    print(f"Loading grid config from {args.grid_config}...")
    config = load_config(args.grid_config)
    
    # Extract grid parameters
    nx = config.get('nx', 20)
    ny = config.get('ny', 20)
    kx = config.get('kx', 1.0)
    ky = config.get('ky', 1.0)
    m = config.get('m', 1.0)
    
    mass_map = None
    
    if args.data:
        print(f"Loading data from {args.data}...")
        try:
            data = np.load(args.data)
            if data.shape == (ny, nx):
                mass_map = data
            else:
                print(f"Data shape {data.shape} does not match grid {ny}x{nx}")
                return 1
        except Exception as e:
            print(f"Error loading data: {e}")
            return 1
            
    # Build grid
    grid = OscillatorGrid2D(
        nx=nx, ny=ny, kx=kx, ky=ky, m=m,
        mass_map=mass_map
    )
    
    # Calculate LDOS
    print("Calculating current LDOS...")
    n_modes = config.get('n_modes', 50)
    ldos_current = grid.ldos_map(n_modes=n_modes, freq_window=profile.freq_window)
    
    # Score
    print("Scoring state...")
    scores = score_ndt_state(profile, ldos_current)
    
    # Threshold
    threshold = config.get('threshold', 3.0)
    mask = ndt_defect_mask(scores, threshold)
    
    n_defects = np.sum(mask)
    print(f"Defects detected: {n_defects} pixels (Threshold={threshold})")
    
    if args.report:
        from .report import generate_ndt_report
        print(f"Generating report: {args.report}")
        generate_ndt_report(
            profile_ldos=profile.ldos_mean,
            current_ldos=ldos_current,
            scores=scores,
            mask=mask,
            out_path=args.report
        )
        
    return 1 if n_defects > 0 else 0


def geo2d_train_command(args):
    """Execute Geo2D training command."""
    print(f"Loading geo config from {args.geo_config}...")
    config = load_config(args.geo_config)
    
    nx = config.get('nx', 30)
    ny = config.get('ny', 30)
    depth_scale = config.get('depth_scale', 1.0)
    
    # Simple homogeneous model for training
    stiffness = config.get('stiffness', 10.0)
    density = config.get('density', 3.0)
    
    stiffness_map = np.full((ny, nx), stiffness)
    density_map = np.full((ny, nx), density)
    
    print(f"Building GeoGrid2D ({nx}x{ny})...")
    geo_grid = GeoGrid2D(
        nx=nx, ny=ny, depth_scale=depth_scale,
        stiffness_map=stiffness_map,
        density_map=density_map
    )
    
    # Calculate surface response (profile)
    freq_window = tuple(config.get('freq_window', [0.0, 2.0]))
    n_modes = config.get('n_modes', 50)
    
    print("Calculating surface response...")
    response = geo_grid.forward_response(freq_window=freq_window, n_modes=n_modes)
    
    print(f"Saving profile to {args.out}...")
    np.savez(args.out, response=response, freq_window=freq_window)
    print("Done.")
    return 0


def geo2d_scan_command(args):
    """Execute Geo2D scan command."""
    print("Geo2D Scan: Not fully implemented yet.")
    print("Use examples/geophysics_2d_toy_demo.py for demonstration.")
    return 0


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
    
    # NDT Train
    parser_ndt_train = subparsers.add_parser('ndt-train', help='Train NDT profile')
    parser_ndt_train.add_argument('--grid-config', required=True, help='Path to grid config.yaml')
    parser_ndt_train.add_argument('--profile-out', required=True, help='Output path for profile.npz')
    parser_ndt_train.set_defaults(func=ndt_train_command)
    
    # NDT Score
    parser_ndt_score = subparsers.add_parser('ndt-score', help='Score NDT state')
    parser_ndt_score.add_argument('--grid-config', required=True, help='Path to grid config.yaml')
    parser_ndt_score.add_argument('--profile', required=True, help='Path to profile.npz')
    parser_ndt_score.add_argument('--data', help='Path to data.npy (mass map)')
    parser_ndt_score.add_argument('--report', help='Path to output report')
    parser_ndt_score.set_defaults(func=ndt_score_command)

    # Geo2D Train
    parser_geo_train = subparsers.add_parser('geo2d-train', help='Train Geo2D profile')
    parser_geo_train.add_argument('--geo-config', required=True, help='Path to geo config.yaml')
    parser_geo_train.add_argument('--out', required=True, help='Output path for profile.npz')
    parser_geo_train.set_defaults(func=geo2d_train_command)
    
    # Geo2D Scan
    parser_geo_scan = subparsers.add_parser('geo2d-scan', help='Scan Geo2D')
    parser_geo_scan.set_defaults(func=geo2d_scan_command)
    
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
