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

        dt = np.mean(np.diff(time))
        N = len(time)
        freqs = fftfreq(N, dt)
        target_freq_hz = omega_drive_rads
        

        idx = np.argmin(np.abs(freqs - target_freq_hz))
        

        amp_phase = np.angle(fft(amp)[idx])
        drive_phase = np.angle(fft(drive)[idx])
        
        #HOLY THIS TOOK WAY TOO LONG TO FIGURE OUT 
        phase_diff = np.angle(np.exp(1j * ((drive_phase - amp_phase) - np.pi)))

        

        phase_data.append((target_freq_hz, phase_diff))
    
    return np.array(phase_data)


df = pd.read_csv('/Users/willgray/Downloads/part-2-data-12trails.csv')


times = []
amps = []
drives = []

for run in range(1, 12): 
    time_col = f'Time (s) Run #{run}'
    chA_col = f'Voltage, Ch A (V) Run #{run}'
    chB_col = f'Voltage, Ch B (V) Run #{run}'
    
    times.append(df[time_col].values)
    amps.append(df[chA_col].values)
    drives.append(df[chB_col].values)

# our driving force 
omega_drives_hz = np.array([0.36,0.46,0.56,0.66,0.76,0.86,0.96,1.06,1.16,1.36,1.26])



phase_points = calculate_phase_points(omega_drives_rads, times, amps, drives)


print("\nPhase Shift Data Points:")
print(f"{'Trial':<6} {'Frequency (Hz)':<15} {'Phase Shift (rad)':<15}")
for i, (freq_hz, phase) in enumerate(phase_points):
    print(f"{i+1:<6} {freq_hz:<15.3f} {phase:<15.3f}")

# Plot!!!
plt.figure(figsize=(8, 5))
plt.scatter(phase_points[:, 0], phase_points[:, 1], c='red', s=50, edgecolor='black')

plt.title('Phase Shift vs Driving Frequency')
plt.xlabel('Driving Frequency (Hz)')
plt.ylabel('Phaxse Shift δ (radians)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
