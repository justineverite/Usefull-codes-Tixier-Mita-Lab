import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- TO BE USED WITH CSV EXTRACTED ANYTRACK FILES ---

# Import your CSV file path
csv_file = ("my_file.csv")

data = pd.read_csv(csv_file)
time = data['x0000'].values #works only with CSV anytrack extracted files, change name of the columns if necessary
displacement = data['y0000'].values

displacement = displacement - np.mean(displacement)
dt = np.mean(np.diff(time))
Fs = 1 / (dt)
N = len(displacement)

# Hanning window (bore effect)
window = np.hanning(N)
displacement_windowed = displacement * window

# FFT calculation
fft_values = np.fft.fft(displacement_windowed)
frequencies = np.fft.fftfreq(N, d=dt)

# Positive frequency and exclusion of 0Hz frequency
half_N = N // 2
frequencies = frequencies[1:half_N]
amplitude = np.abs(fft_values[1:half_N]) * 2 / N  # Normalisation


# Ploting the original signal
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(time, displacement, label='Original Signal')
plt.title('Cardiomyocytes Displacement Optical Records (Basal Conditions) - 8 days old culture') # adjust the legend
plt.xlabel('Time (s)')
plt.ylabel('Displacement (pixels)')
plt.legend()
plt.grid(True)

# Ploting the FFT
plt.subplot(2, 1, 2)
plt.plot(frequencies, amplitude, color='red', label='Frequency Spectrum')
plt.title('Fourier Decomposition of Beating Pattern - 8 days old cardiomyocytes') # adjust the legend
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Dominant frequency identification
dominant_freq = frequencies[np.argmax(amplitude)]
print(f" Dominanyty frequency : {dominant_freq} Hz with an amplitude of  {np.max(amplitude)} µV")