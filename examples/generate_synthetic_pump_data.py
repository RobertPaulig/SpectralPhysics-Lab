import numpy as np
import pandas as pd
from pathlib import Path

def generate_signal(t, freqs, amps, noise_level=0.1):
    signal = np.zeros_like(t)
    for f, a in zip(freqs, amps):
        signal += a * np.sin(2 * np.pi * f * t)
    noise = np.random.normal(0, noise_level, size=len(t))
    return signal + noise

def main():
    # Setup
    data_dir = Path("data/pump")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    dt = 0.001
    duration = 10.0
    t = np.arange(0, duration, dt)
    
    print(f"Generating data in {data_dir}...")
    
    # Define "Healthy" characteristics
    # Motor: 50Hz (main), 100Hz (harmonic)
    motor_freqs = [50.0, 100.0]
    motor_amps = [1.0, 0.2]
    
    # Pump: 30Hz (vane pass), 60Hz
    pump_freqs = [30.0, 60.0]
    pump_amps = [0.8, 0.3]
    
    # 1. Generate Training Data (Healthy)
    # We generate slightly different variations for training
    
    # Train Motor 1
    m1 = generate_signal(t, motor_freqs, motor_amps, noise_level=0.1)
    p1 = generate_signal(t, pump_freqs, pump_amps, noise_level=0.1)
    df1 = pd.DataFrame({'time': t, 'motor_vibration': m1, 'pump_vibration': p1})
    df1.to_csv(data_dir / "train_motor_1.csv", index=False)
    # Note: We save the same file for pump training or different ones?
    # The config expects:
    # motor_vibration files: train_motor_1.csv, train_motor_2.csv
    # pump_vibration files: train_pump_1.csv, train_pump_2.csv
    # We can just reuse the same structure or make separate files if we want strictly separate sensors.
    # But the config implies we might have different files for different channels or same files.
    # Let's generate 4 files as requested by the config structure, 
    # but actually the config says:
    # motor: train_motor_1.csv, train_motor_2.csv
    # pump: train_pump_1.csv, train_pump_2.csv
    # So we should probably generate these 4 files.
    
    # Train Motor 2 (slight variation)
    m2 = generate_signal(t, motor_freqs, [a * 1.05 for a in motor_amps], noise_level=0.12)
    p2 = generate_signal(t, pump_freqs, [a * 0.95 for a in pump_amps], noise_level=0.11)
    df2 = pd.DataFrame({'time': t, 'motor_vibration': m2, 'pump_vibration': p2})
    df2.to_csv(data_dir / "train_motor_2.csv", index=False)
    
    # Train Pump 1 (can be same as motor 1 or different)
    # Let's make them distinct files to match the config exactly
    df1.to_csv(data_dir / "train_pump_1.csv", index=False)
    df2.to_csv(data_dir / "train_pump_2.csv", index=False)
    
    print("  Created training files.")
    
    # 2. Generate Current Data (Test)
    
    # Current Motor (Normal)
    m_curr = generate_signal(t, motor_freqs, [a * 1.02 for a in motor_amps], noise_level=0.1)
    p_curr_normal = generate_signal(t, pump_freqs, pump_amps, noise_level=0.1)
    df_curr_motor = pd.DataFrame({'time': t, 'motor_vibration': m_curr, 'pump_vibration': p_curr_normal})
    df_curr_motor.to_csv(data_dir / "current_motor.csv", index=False)
    
    # Current Pump (Anomaly!)
    # Add a new frequency component at 150Hz (bearing fault?) and increase noise
    m_anom = generate_signal(t, motor_freqs, motor_amps, noise_level=0.1)
    
    pump_freqs_anom = pump_freqs + [150.0]
    pump_amps_anom = pump_amps + [0.5] # Significant new peak
    p_anom = generate_signal(t, pump_freqs_anom, pump_amps_anom, noise_level=0.2) # Higher noise
    
    df_curr_pump = pd.DataFrame({'time': t, 'motor_vibration': m_anom, 'pump_vibration': p_anom})
    df_curr_pump.to_csv(data_dir / "current_pump.csv", index=False)
    
    print("  Created current status files (Motor: OK, Pump: ANOMALY).")
    print("Done.")

if __name__ == "__main__":
    main()
