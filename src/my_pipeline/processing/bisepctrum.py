import logging
import numpy as np
from my_pipeline.processing.constants import (
    TOTAL_CHANNEL_PAIRS, 
    STANDARD_EEG_CHANNELS, 
    SMOOTHING_WINDOW,
    FREQUENCY_TRIPLE_INTERESTED_BY_BAND_PAIR)
from scipy.signal import convolve

logger = logging.getLogger(__name__)

def calculate_instant_bispectrum(W1, W2, W3):
    """
    Calculate wavelet bispectrum time series for frequency pair (f1, f2)
    
    The bispectrum at each time point is:
    B(f1, f2, t) = W(f1, t) * W(f2, t) * conj(W(f1+f2, t))
    
    Parameters:
    - W1, W2, W3: complex wavelet coefficients (1D arrays)
    
    Returns:
    - dict: biamplitude
    """
    bispectrum_complex = W1 * W2 * np.conj(W3)
    
    biamplitude = np.abs(bispectrum_complex)
    
    return {
        'biamplitude': biamplitude,
    }
    
def calculate_smoothed_bispectrum(W1, W2, W3, smoothing_window):
    """
    Calculate smoothed wavelet bispectrum, biamplitude, bispectral phase, and bicoherence.

    All calculations are based on a time smoothing window (E[...]).

    Parameters:
    - W1 (array): frequency f1's complex wavelet coefficients (1D array, time series).
    - W2 (array): frequency f2's complex wavelet coefficients (1D array, time series).
    - W3 (array): and frequency f3=f1+f2's complex wavelet coefficients (1D array, time series).
    - smoothing_window (array): 1D window for smoothing (e.g., a normalized Hann window).

    Returns:
    - dict: 
        'bispectrum_complex': E[ W1 * W2 * conj(W3) ] 
        'biamplitude': | E[...] |
        'bicoherence': normalized bicoherence (0-1)
    """

    # --- 1. Calculate instantaneous bispectrum ---
    B_inst = W1 * W2 * np.conj(W3)

    # --- 2. Calculate smoothed complex bispectrum (E[B_inst]) ---
    B_W_real = convolve(np.real(B_inst), smoothing_window, mode='same')
    B_W_imag = convolve(np.imag(B_inst), smoothing_window, mode='same')
    
    # E[ B_inst ]
    bispectrum_smoothed = B_W_real + 1j * B_W_imag
    
    # --- 3. Calculate bicoherence ---
    numerator = np.abs(bispectrum_smoothed)**2

    # Denominator: E[ |W1 * W2|^2 ] * E[ |W3|^2 ]
    
    # First term: E[ |W(f1) * W(f2)|^2 ]
    den1 = convolve(np.abs(W1 * W2)**2, smoothing_window, mode='same')
    
    # Second term: E[ |W(f3)|^2 ]
    den2 = convolve(np.abs(W3)**2, smoothing_window, mode='same')

    denominator = den1 * den2
    
    epsilon = 1e-10 # To avoid division by zero
    bicoherence_squared = numerator / (denominator + epsilon)
    
    # Final bicoherence (b)
    bicoherence = np.sqrt(bicoherence_squared)
    
    # --- 4. Derive other values from smoothed bispectrum ---
    biamplitude_smoothed = np.abs(bispectrum_smoothed)

    
    return {
        'biamplitude': biamplitude_smoothed,
        'bicoherence': bicoherence
    }

def run(cwt_results, segment_id, smoothed):
    """Compute smoothed bispectrum for all channel pairs."""
    logging.info("Computing smoothed bispectrum for all channel pairs for segment %s", segment_id)
    if not smoothed:
        logger.warning("Bispectrum calculation is set to INSTANT (not smoothed). Bicoherence will not be computed.")
    
    bispectrum_results = {}
    channel_pair_progress = 0
    
    for idx_ch1, ch_name1 in enumerate(STANDARD_EEG_CHANNELS):
        for idx_ch2 in range(idx_ch1 + 1, len(STANDARD_EEG_CHANNELS)):
            ch_name2 = STANDARD_EEG_CHANNELS[idx_ch2]
            cwt_coefficients_1 = cwt_results[idx_ch1]
            cwt_coefficients_2 = cwt_results[idx_ch2]
            channel_pair = (idx_ch1, idx_ch2)
            bispectrum_results[channel_pair] = {}
            logging.info(f"Calculating for channel pair: {ch_name1}, {ch_name2} ({channel_pair_progress+1}/{TOTAL_CHANNEL_PAIRS})")
            
            for band_pair, freq_triples in FREQUENCY_TRIPLE_INTERESTED_BY_BAND_PAIR.items():
                n_freq_pairs, avg_biamplitude, avg_bicoherence = 0, 0.0, 0.0
                
                for freq_idx1, freq_idx2, freq_idx3 in freq_triples:
                    W1 = cwt_coefficients_1[freq_idx1, :]
                    W2 = cwt_coefficients_2[freq_idx2, :]
                    W3 = cwt_coefficients_2[freq_idx3, :]
                    
                    if smoothed:
                        bispec_result = calculate_smoothed_bispectrum(W1, W2, W3, SMOOTHING_WINDOW)
                    else:
                        bispec_result = calculate_instant_bispectrum(W1, W2, W3)
                    
                    if bispec_result is not None:
                            n_freq_pairs += 1
                            
                            new_biamplitude = bispec_result['biamplitude']
                            if smoothed:
                                new_bicoherence = bispec_result['bicoherence']

                            # Update the average biamplitude using the online formula
                            # avg = old_avg + (new_value - old_avg) / count
                            avg_biamplitude += (new_biamplitude - avg_biamplitude) / n_freq_pairs
                            
                            # Update the average bicoherence similarly
                            if smoothed:
                                avg_bicoherence += (new_bicoherence - avg_bicoherence) / n_freq_pairs
                
                bispectrum_results[channel_pair][band_pair] = {
                    'biamplitude': avg_biamplitude,
                    'bicoherence': avg_bicoherence,
                }
            
            channel_pair_progress += 1

            # Check if channel 2 is the last channel
            if idx_ch2 == len(STANDARD_EEG_CHANNELS) - 1:
                logging.info(f"Completed all pairs for channel {ch_name1} ({idx_ch1+1}/{len(STANDARD_EEG_CHANNELS)})")
                
                # Release memory for channel 1's CWT results
                del cwt_results[idx_ch1]
                
    return bispectrum_results
    
