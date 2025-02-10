import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt, find_peaks


# Load file here (.txt)
data = np.loadtxt("My_file.txt", skiprows = 3, max_rows=100000) #skiprows to be adjusted (size of the header), max_rows to be adjusted (zoom)
time = data[:,0]
signal = data[:,1] #to be changed depending on the column (electrode) considered


# Smoothing, filtering (band-pass filter) the signal and spike detection

signal_filtre = medfilt(signal, kernel_size=5) #to be adjusted

# --- Spike detector parameter (threshold) ---
seuil = 110
indices_pics = np.where(np.abs(signal) > seuil)[0]

# --- Low pass filter parameters (fc and order to be adjusted) ---
fc = 3.5 #adjust
order = 4 #adjust
fs = 1000/np.mean(np.diff(time)) #for 10000 Hz measurements
nyquist = fs/2
fc_normalise = fc/nyquist
b,a = butter(order, fc_normalise, btype="low", analog=False)
signal_filtre2 = filtfilt(b, a, signal)

# --- High pass filter parameters (fc and order to be adjusted) ---
fc = 1
order = 4
fs = 1000/np.mean(np.diff(time))
nyquist = fs/2
fc_normalise = fc/nyquist
d,c = butter(order, fc_normalise, btype="high", analog=False)

# signal filtré
signal_filtre3 = filtfilt(d, c, signal_filtre2)
signal_lisse_avec_pics = np.copy(signal_filtre3)
signal_lisse_avec_pics[indices_pics] = signal[indices_pics]


# Plotting the signal
plt.figure(figsize =(20,5))
# plt.plot(time, signal, label = "signal no filter", color = 'green')
plt.plot(time, signal_filtre, label = "Raw signal", color = 'red',)
plt.plot(time, signal_lisse_avec_pics, label = "Filtered signal - Butterworth low pass (CF:30Hz, order 4)", color = 'blue') # !!! parameters to be adjusted !!!
plt.title("Electrophysiology recordings, 3 days old cardiomyocytes, 100nM, immediate rinse - 3 days after NE") # !!! parameters to be adjusted !!!
plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (µV)')
plt.show()



# Mean peaks amplitude calculation (threshold to be adjusted) :


peaks_pos, _ = find_peaks(signal, height=100)  # Threshold to be adjusted
peaks_neg, _ = find_peaks(-signal, height=100)  # Threshold to be adjusted

amplitudes_totales = []
distance_max = 20

for pos in peaks_pos:
    diffs = np.abs(peaks_neg - pos)
    nearest_neg_idx = np.argmin(diffs)
    nearest_neg = peaks_neg[nearest_neg_idx]

    if diffs[nearest_neg_idx] <= distance_max:
        amplitude_positive = signal[pos]
        amplitude_negative = signal[nearest_neg]
        amplitude_totale = amplitude_positive - amplitude_negative
        amplitudes_totales.append(amplitude_totale)

if amplitudes_totales:
    moyenne_amplitudes = np.mean(amplitudes_totales)
    print(f"Mean of amplitudes (positive and negative) : {moyenne_amplitudes:.2f} mV")
else:
    print("No spikes detected.")


# Creation of a txt file containing smoothed signal data !!!!

file = open("courbe_lisse.txt", "w")
for e in signal_lisse_avec_pics:
    file.write(str(e))
    file.write('\n')
file.close()
