import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# File opening and data extraction (.txt files)
def read_data_with_headers(file_path):
    try:
        data = np.loadtxt(file_path, delimiter='\t', skiprows= 1)  # Skip rows can be adjusted depending on the size of the header
        x, y = data[:, 0], data[:, 1]
        return x, y
    except Exception as e:
        print(f"File reading error : {e}")
        return None, None



# Graph tracing
def plot_data(x, y):
    plt.figure(figsize=(20, 3)) #can be adjusted depending on the size of the measurement
    plt.plot(x, y, color='b')
    plt.title("Electrophysiology Measurement of Beating Cardiomyocites (Field Potential)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (µV)")
    plt.grid(True)
    plt.legend()
    plt.show()



# File information and reading
file_path = ("My_file.txt") #input file path here
x, y = read_data_with_headers(file_path)
if x is not None and y is not None:
    plot_data(x, y)



# Smoothing and spike sorting function
def smooth(y, window_length=50, polyorder=3, threshold = 60): #parameters to be changed depending on the signal (threshold : spike sorting threshold)
    y_smooth = y.copy()
    non_peak_indices = np.abs(y) < threshold
    # Spike sorting
    y_smooth[non_peak_indices] = savgol_filter(y[non_peak_indices], window_length=window_length, polyorder=polyorder)
    return y_smooth


# Example without spike sorting
y_smooth = smooth(y, window_length=220, polyorder=3)


# Plot
plt.figure(figsize=(20, 3))
plt.plot(x, y, color='b', linestyle='dashed', linewidth= 0.5)
plt.plot(x[:len(y_smooth)], y_smooth, label='Smoothed signal (Savitzky-Golay, w=30, po=3 )', color='red') # !! Change parameters !!
plt.title('Smoothed Electrophysiology Field Potential Signal of Beating Cardiomyocites')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (µV)')
plt.legend()
plt.grid(True)
plt.show()