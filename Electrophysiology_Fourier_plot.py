import numpy as np

import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Charger les données depuis un fichier CSV
time = np.loadtxt("C:/Users/Justine/Desktop/experiment1_day3_imediate/txt/after_filtre.txt",skiprows=10002, max_rows=180000)
t = time[:,0]  # Temps en millisecondes

#choose between the two lines below (depending on the situation : raw data or previously filtered data) :

#signal = np.loadtxt("courbe_lisse.txt", skiprows=10000, max_rows=180000) #!!!! to be used after running "Electrophysiology_plotting_bandpass_filter.py" on raw data!!!
#signal = time[:,1] #to be used on non filtered (raw) data



# Converting time in seconds (ms to s)
t = t / 1000.0


# Fourier calculation
N = len(t)
T = t[1] - t[0]
yf = fft(signal)
xf = fftfreq(N, T)[:N//2]  # positive frequencies
# Exclusion of 0Hz and upper limit (can be adjusted)
non_zero_indices = (xf > 0) & (xf <= 200)
xf = xf[non_zero_indices]
amplitude_spectre = (2.0 / N * np.abs(yf[:N//2]))[non_zero_indices]

# Dominant frequency (max amplitude)
frequence_dominante = xf[np.argmax(amplitude_spectre)]

# Plot showing : fourier transform
plt.figure(figsize=(8, 5))
plt.plot(xf, amplitude_spectre)
plt.title("Fourier transform on cell electrophysiology signal 5 min after adding \n norepinephrine, 0-18 seconds (0-100 Hz) ") # text to be adjusted
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude ")
plt.grid()
plt.xlim(0, 50)  # Limit of X axis : can be adjusted
plt.show()

# Printing dominant frequency
print(f"Fréquence dominante détectée : {frequence_dominante:.2f} Hz")

# plot showing : electrophysiology signal
t2 = t*1000
plt.figure(figsize=(10, 5))
plt.plot(t2, signal)
plt.title("Zoom on spikes, filtered signal, before norepinephrine")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (mV) ")
plt.grid()
plt.show()