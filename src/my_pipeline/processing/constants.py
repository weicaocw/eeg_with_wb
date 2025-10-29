from collections import defaultdict

import numpy as np
from scipy.signal import windows
import matplotlib.pyplot as plt

SAMPLING_RATE = 250
FREQUENCY_BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8), 
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 100)
}
BAND_NAMES = list( FREQUENCY_BANDS.keys())
BAND_PAIRS = []
for i in range(len(BAND_NAMES)):
    for j in range(i, len(BAND_NAMES)):
        BAND_PAIRS.append( (BAND_NAMES[i], BAND_NAMES[j]) )

STANDARD_EEG_CHANNELS = [
    'FP1', 
    'FP2', 
    'F3', 
    'F4', 
    'C3', 
    'C4', 
    'P3', 
    'P4', 
    'O1', 
    'O2', 
    'F7', 
    'F8', 
    'T3', 
    'T4', 
    'T5', 
    'T6', 
    'CZ']

TOTAL_CHANNEL_PAIRS = len(STANDARD_EEG_CHANNELS) * (len(STANDARD_EEG_CHANNELS) - 1) // 2

DEFAULT_CWT_WAVELET = 'cmor1.5-1.0'

FREQUENCY_USED = np.logspace(np.log10(0.5), np.log10(100), 100)  # 100 frequency points

WINDOW_DURATION_SEC = 1.0 
WINDOW_SAMPLES = int(SAMPLING_RATE * WINDOW_DURATION_SEC)
SMOOTHING_WINDOW = windows.hann(WINDOW_SAMPLES) #TODO Configurable smoothing window
SMOOTHING_WINDOW /= np.sum(SMOOTHING_WINDOW)

FREQUENCY_TRIPLE_INTERESTED_BY_BAND_PAIR = defaultdict(list)
for freq_idx in range(len(FREQUENCY_USED)):
    for freq_idx2 in range(freq_idx, len(FREQUENCY_USED)):
        f1 = FREQUENCY_USED[freq_idx]
        f2 = FREQUENCY_USED[freq_idx2]
        f3 = f1 + f2
        if f3 >= FREQUENCY_USED[-1]:
            continue
        freq_idx3 = np.argmin(np.abs(FREQUENCY_USED - f3))
        
        band_name1 = None
        band_name2 = None
        for band_name, (f_min, f_max) in FREQUENCY_BANDS.items():
            if f_min <= f1 < f_max:
                band_name1 = band_name
            if f_min <= f2 < f_max:
                band_name2 = band_name
        if band_name1 is None or band_name2 is None:
            continue
        FREQUENCY_TRIPLE_INTERESTED_BY_BAND_PAIR[(band_name1, band_name2)].append((freq_idx, freq_idx2, freq_idx3))

CMAP = plt.get_cmap('Set2')

EVENT_LABEL_BACKGROUND = 'bckg'