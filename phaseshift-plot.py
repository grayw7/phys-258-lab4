import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq

def calculate_phase_points(omega_drives_rads, all_times, all_amps, all_drives):
    phase_data = []
    
    for trial_num, (omega_drive_rads, time, amp, drive) in enumerate(zip(
        omega_drives_rads, all_times, all_amps, all_drives)):
        
        # Clean data
        valid_mask = ~(np.isnan(time) | np.isnan(amp) | np.isnan(drive))
        time = time[valid_mask]
        amp = amp[valid_mask]
        drive = drive[valid_mask]
        
        if len(time) < 2:
            print(f"Skipping Trial {trial_num+1} - insufficient data")
            continue

        # Calculate FFT parameters
        dt = np.mean(np.diff(time))
        N = len(time)
        freqs = fftfreq(N, dt)  # Frequencies in Hz
        target_freq_hz = omega_drive_rads
        
        # Find nearest frequency bin
        idx = np.argmin(np.abs(freqs - target_freq_hz))
        
        # Calculate phases
        amp_phase = np.angle(fft(amp)[idx])
        drive_phase = np.angle(fft(drive)[idx])
        
        # Calculate phase difference (0 to 2π range)
        phase_diff = np.unwrap([amp_phase - drive_phase])[0] % (2 * np.pi)

        
        # Store results
        phase_data.append((target_freq_hz, phase_diff))
    
    return np.array(phase_data)

# Load data
df = pd.read_csv('/Users/willgray/Downloads/part-2-data-12trails.csv')

# Extract data for 10 trials
num_trials = 10
times = []
amps = []
drives = []

for run in range(1, 12):  # Runs 2-11 (10 trials)
    time_col = f'Time (s) Run #{run}'
    chA_col = f'Voltage, Ch A (V) Run #{run}'
    chB_col = f'Voltage, Ch B (V) Run #{run}'
    
    times.append(df[time_col].values)
    amps.append(df[chA_col].values)
    drives.append(df[chB_col].values)

# Driving frequencies in rad/s (from your example)
omega_drives_rads = np.array([0.36,0.46,0.56,0.66,0.76,0.86,0.96,1.06,1.16,1.26])

# Calculate phase points
phase_points = calculate_phase_points(omega_drives_rads, times, amps, drives)

# Print all points
print("\nPhase Shift Data Points:")
print(f"{'Trial':<6} {'Frequency (Hz)':<15} {'Phase Shift (rad)':<15}")
for i, (freq_hz, phase) in enumerate(phase_points):
    print(f"{i+1:<6} {freq_hz:<15.3f} {phase:<15.3f}")

# Plot results
plt.figure(figsize=(8, 5))
plt.scatter(phase_points[:, 0], phase_points[:, 1], c='red', s=50, edgecolor='black')

plt.title('Phase Shift vs Driving Frequency')
plt.xlabel('Driving Frequency (Hz)')
plt.ylabel('Phase Shift δ (radians)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
