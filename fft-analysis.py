import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit


# -----------------------------------------------------------
# Model Functions for Curve Fitting
# -----------------------------------------------------------
def phase_shift_model(omega, omega_0, beta):
    
    numerator = omega_0**2 - omega**2
    denominator = np.sqrt((omega_0**2 - omega**2)**2 + 4 * omega**2 * beta**2)
    cos_delta = numerator / denominator
    return np.arccos(cos_delta)


def lorentzian(omega, A0, omega0, gamma):
    #Lorentzian function used for amplitude fitting.
    
    return A0 / np.sqrt((omega0**2 - omega**2)**2 + (gamma * omega)**2)


# -----------------------------------------------------------
# Data Extraction Functions (using FFT)
# -----------------------------------------------------------
def calculate_phase_points(omega_drives, all_times, all_amps, all_drives):
    #Calculate phase shift data points for each trial.

    phase_data = []
    
    for trial_num, (omega_drive, time, amp, drive) in enumerate(zip(
            omega_drives, all_times, all_amps, all_drives)):
        

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
        

        idx = np.argmin(np.abs(freqs - omega_drive))
        
        amp_phase = np.angle(fft(amp)[idx])
        drive_phase = np.angle(fft(drive)[idx])
        
        # Compute phase difference (with π-shift correction) SO IMPORTANT 
        phase_diff = np.angle(np.exp(1j * ((drive_phase - amp_phase) - np.pi)))
        phase_data.append((omega_drive, phase_diff))
    
    return np.array(phase_data)


def calculate_amplitude_points(omega_drives, all_times, all_amps):
    #Calculate amplitude data points for each trial.

    amplitude_data = []
    
    for trial_num, (omega_drive, time, amp) in enumerate(zip(omega_drives, all_times, all_amps)):
        # Clean data
        valid_mask = ~(np.isnan(time) | np.isnan(amp))
        time = time[valid_mask]
        amp = amp[valid_mask]
        
        if len(time) < 2:
            print(f"Skipping Trial {trial_num+1} for amplitude - insufficient data")
            continue

        dt = np.mean(np.diff(time))
        N = len(time)
        freqs = fftfreq(N, dt)
        
        # Find index of the driving frequency
        idx = np.argmin(np.abs(freqs - omega_drive))
        
        amp_magnitude = np.abs(fft(amp)[idx])
        amplitude_data.append((omega_drive, amp_magnitude))
        
    return np.array(amplitude_data)


# -----------------------------------------------------------
# Data Loading
# -----------------------------------------------------------
def load_data(file_path):
    
    df = pd.read_csv(file_path)
    
    times, amps, drives = [], [], []
    # We had 11 trials; adjust if necessary.
    for run in range(1, 12): 
        time_col = f'Time (s) Run #{run}'
        chA_col = f'Voltage, Ch A (V) Run #{run}'
        chB_col = f'Voltage, Ch B (V) Run #{run}'
        
        times.append(df[time_col].values)
        amps.append(df[chA_col].values)
        drives.append(df[chB_col].values)
    
    # Our driving frequencies (in Hz) hard-coded (adjust if needed)
    omega_drives = np.array([0.36, 0.46, 0.56, 0.66, 0.76, 0.86, 0.96, 1.06, 1.16, 1.36, 1.26])
    
    return omega_drives, times, amps, drives


# -----------------------------------------------------------
# Fitting Functions
# -----------------------------------------------------------
def fit_phase_shift(phase_points, p0_phase=[1.0, 0.1]):
    
    popt, pcov = curve_fit(phase_shift_model, phase_points[:, 0], phase_points[:, 1], p0=p0_phase)
    return popt, pcov


def fit_amplitude(amplitude_points):

    max_index = np.argmax(amplitude_points[:, 1])
    A0_guess = amplitude_points[max_index, 1]
    omega0_guess = amplitude_points[max_index, 0]
    p0_amp = [A0_guess, omega0_guess, 0.1]
    
    popt, pcov = curve_fit(lorentzian, amplitude_points[:, 0], amplitude_points[:, 1], p0=p0_amp)
    return popt, pcov


# -----------------------------------------------------------
# Plotting Functions
# -----------------------------------------------------------
def plot_phase_shift(phase_points, popt_phase, omega_drives):

    omega_fit = np.linspace(np.min(omega_drives), np.max(omega_drives), 10000)
    
    plt.figure(figsize=(8, 5))
    plt.scatter(phase_points[:, 0], phase_points[:, 1],
                c='purple', s=50, label='Data',zorder=2)
    plt.plot(omega_fit, phase_shift_model(omega_fit, *popt_phase)
             ,color = "green", lw=2, label='Fit', zorder=1)
    plt.title('Phase Shift vs Driving Frequency')
    plt.xlabel('Driving Frequency (Hz)')
    plt.ylabel('Phase Shift δ (radians)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_amplitude(amplitude_points, popt_amp, omega_drives):

    omega_fit = np.linspace(np.min(omega_drives), np.max(omega_drives), 10000)
    
    plt.figure(figsize=(8, 5))
    plt.scatter(amplitude_points[:, 0], amplitude_points[:, 1],
                c='purple', s=50, label='Data', zorder=2)
    plt.plot(omega_fit, lorentzian(omega_fit, *popt_amp), color = "green", lw=2, label='Fit', zorder=1)
    plt.title('Amplitude vs Driving Frequency')
    plt.xlabel('Driving Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------
# Print Results Function
# -----------------------------------------------------------
def print_results(phase_points, amplitude_points, popt_phase, popt_amp):
    """
    Print the data points and the fitted parameters.
    """
    omega_0_fit, beta_fit = popt_phase
    A0_fit, omega0_fit, gamma_fit = popt_amp

    print("\nPhase Shift Data Points:")
    print(f"{'Trial':<6} {'Frequency (Hz)':<15} {'Phase Shift (rad)':<15}")
    for i, (freq, phase) in enumerate(phase_points):
        print(f"{i+1:<6} {freq:<15.3f} {phase:<15.3f}")

    print("\nFitted Phase Shift Parameters:")
    print(f"omega_0 = {omega_0_fit:.3f} Hz, beta = {beta_fit:.3f}")

    print("\nAmplitude Data Points:")
    print(f"{'Trial':<6} {'Frequency (Hz)':<15} {'Amplitude':<15}")
    for i, (freq, amp_val) in enumerate(amplitude_points):
        print(f"{i+1:<6} {freq:<15.3f} {amp_val:<15.3f}")

    print("\nFitted Amplitude (Lorentzian) Parameters:")
    print(f"A0 = {A0_fit:.3f}, omega0 = {omega0_fit:.3f} Hz, gamma = {gamma_fit:.3f}")


# -----------------------------------------------------------
# Main Function
# -----------------------------------------------------------
def main():
    # Specify the file path to your CSV data file.
    file_path = '/Users/willgray/Downloads/part-2-data-12trails.csv'
    
    # Load the experimental data
    omega_drives, times, amps, drives = load_data(file_path)
    
    # Compute the FFT-based data points
    phase_points = calculate_phase_points(omega_drives, times, amps, drives)
    amplitude_points = calculate_amplitude_points(omega_drives, times, amps)
    
    # Fit the data to the corresponding models
    popt_phase, _ = fit_phase_shift(phase_points)
    popt_amp, _ = fit_amplitude(amplitude_points)
    
    # Print the results
    print_results(phase_points, amplitude_points, popt_phase, popt_amp)
    
    # Plot the results
    plot_phase_shift(phase_points, popt_phase, omega_drives)
    plot_amplitude(amplitude_points, popt_amp, omega_drives)


if __name__ == '__main__':
    main()
