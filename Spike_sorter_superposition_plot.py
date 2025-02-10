import matplotlib.pyplot as plt

# --- TO BE USED WITH SPIKE SORTER FUNCTION FROM MCRACK EXTRACTED TXT FILES ---

# Load txt (spike sorter from mcrack)
fichier = "My_file.txt"  # Remplacez par le nom de votre fichier
with open(fichier, 'r') as f:
    lignes = f.readlines()


all_timelines = []
all_amplitudes = []

timelines = []
amplitudes = []


for ligne in lignes:
    if ligne.strip():  # if not empty
        valeurs = ligne.split()
        timeline = float(valeurs[0])  # Timeline in ms
        amplitude = float(valeurs[1])  # Amplitude
        timelines.append(timeline)
        amplitudes.append(amplitude)
    else:
        # if empty
        if timelines:
            all_timelines.append(timelines)
            all_amplitudes.append(amplitudes)
            timelines = []
            amplitudes = []

# add last points
if timelines:
    all_timelines.append(timelines)
    all_amplitudes.append(amplitudes)

# spikes superposition
plt.figure(figsize=(10, 6))

for i, (timelines, amplitudes) in enumerate(zip(all_timelines, all_amplitudes)):
    # Adjustment of timelines (beginning at 0 ms)
    adjusted_timeline = [t - timelines[0] for t in timelines]
    plt.plot(adjusted_timeline, amplitudes, label=f'Pic {i+1}')

# title and legends
plt.title('Spikes superposition')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()