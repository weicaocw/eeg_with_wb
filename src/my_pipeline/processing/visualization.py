import logging
import os
from collections import defaultdict

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from my_pipeline.processing.utils import (smooth_per_second, 
                                          get_channel_pairs_from_seizure_events,
                                          get_pair_colors,
                                          get_seizure_events,
                                          get_seizure_event_time_windows,
                                          get_seizure_event_time_windows_for_a_channel_pair)
from my_pipeline.processing.constants import (SAMPLING_RATE, 
                                              BAND_NAMES,
                                              BAND_PAIRS,
                                              STANDARD_EEG_CHANNELS)

logger = logging.getLogger(__name__)
    

def visualize_all_channel_pairs(bispectrum_results, 
                                seg_metadata,
                                property_names, 
                                save_file_path=None,
                                smooth_per_sec=True):
    seizure_events = get_seizure_events(seg_metadata)
    channel_pairs_from_seizure_event = get_channel_pairs_from_seizure_events(seizure_events)
    pair_colors = get_pair_colors(channel_pairs_from_seizure_event)
    
    for property_name in property_names:
        _, axes = plt.subplots(5, 5, figsize=(25, 20))
        axes = axes.flatten()
    
        for band_idx1, band_name1 in enumerate(BAND_NAMES):
            for band_idx2, band_name2 in enumerate(BAND_NAMES):
                ax = axes[band_idx1 * len(BAND_NAMES) + band_idx2]
                for ch_idx1, ch_idx2 in channel_pairs_from_seizure_event:
                    ch_idx_pair_key = (ch_idx1, ch_idx2)
                    if ch_idx_pair_key not in bispectrum_results:
                        ch_idx_pair_key = (ch_idx2, ch_idx1)
                    band_pair_key = (band_name1, band_name2)
                    if (band_name1, band_name2) not in bispectrum_results[ch_idx_pair_key]:
                        band_pair_key = (band_name2, band_name1)
                    data = bispectrum_results[ch_idx_pair_key][band_pair_key][property_name]
                    n_samples = len(data)
                    time = np.arange(n_samples) / SAMPLING_RATE
                    if smooth_per_sec:
                        data = smooth_per_second(data, SAMPLING_RATE)
                        time = np.arange(data.shape[-1])
                    ax.plot(time, data, 
                            color=pair_colors[(ch_idx1, ch_idx2)], 
                            alpha=1,
                            label=f"{STANDARD_EEG_CHANNELS[ch_idx1]}-{STANDARD_EEG_CHANNELS[ch_idx2]}")
                ax.set_title(f'{band_name1} - {band_name2}')
                ax.set_ylabel(property_name)
                
                seizure_event_time_windows = get_seizure_event_time_windows(seizure_events)
                for seizure_start, seizure_stop in seizure_event_time_windows:
                    ax.axvspan(seizure_start, seizure_stop, color='red', alpha=0.1)
        plt.legend(bbox_to_anchor=(1.05, 6), loc='upper right', fontsize='small', ncol=1)
        plt.tight_layout()
        
        if save_file_path:
            try:
                suffix = '_smoothed' if smooth_per_sec else ''
                os.makedirs(save_file_path, exist_ok=True)
                output_file_path = os.path.join(save_file_path, f'all_channel_pairs_{property_name}{suffix}-{property_name}.png')
                plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
                logging.info(f"Figure saved to {output_file_path}")
            except Exception as e:
                logging.error(f"Failed to save figure for all channel pairs to {save_file_path}: {e}")
        else:
            logging.Warn("Save file path not provided")
        
        plt.close()
    
def do_visualize_single_channel_pair(bispectrum_results, 
                                  channel_pair,
                                  property_names,
                                  seizure_event_time_windows,
                                  pair_color,
                                  save_file_path=None,
                                  smooth_per_sec=True):
    if channel_pair not in bispectrum_results:
        channel_pair = (channel_pair[1], channel_pair[0])
    
    for property_name in property_names:
        fig, axes = plt.subplots(5, 5, figsize=(20, 15))
        axes = axes.flatten()
        
        for band_idx1, band_name1 in enumerate(BAND_NAMES):
            for band_idx2, band_name2 in enumerate(BAND_NAMES):
                ax = axes[band_idx1 * len(BAND_NAMES) + band_idx2]
                
                band_pair_key = (band_name1, band_name2)
                if (band_name1, band_name2) not in bispectrum_results[channel_pair]:
                    band_pair_key = (band_name2, band_name1)
                    
                data = bispectrum_results[channel_pair][band_pair_key][property_name]
                n_samples = len(data)
                time = np.arange(n_samples) / SAMPLING_RATE
                if smooth_per_sec:
                    data = smooth_per_second(data, SAMPLING_RATE)
                    time = np.arange(data.shape[-1])
                ax.plot(time, data, color=pair_color)
                ax.set_title(f'{band_name1} - {band_name2}')
                ax.set_xlabel('time (s)')
                ax.set_ylabel(property_name)
                
                for seizure_start, seizure_stop in seizure_event_time_windows:
                    ax.axvspan(seizure_start, seizure_stop, color='red', alpha=0.1)
        
        fig.suptitle(f'Channel Pair: {STANDARD_EEG_CHANNELS[channel_pair[0]]} - {STANDARD_EEG_CHANNELS[channel_pair[1]]}', fontsize=16)
        plt.tight_layout()
    
    
        if save_file_path:
            try:
                os.makedirs(save_file_path, exist_ok=True)
                suffix = '_smoothed' if smooth_per_sec else ''
                output_file_path = os.path.join(save_file_path, 
                                            f'{property_name}_{STANDARD_EEG_CHANNELS[channel_pair[0]]}_{STANDARD_EEG_CHANNELS[channel_pair[1]]}{suffix}-{property_name}.png') if save_file_path else None
                plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
                logging.info(f"Figure saved to {output_file_path}")
            except Exception as e:
                logging.error(f"Failed to save figure for channel pair {channel_pair} to {save_file_path}: {e}")
        else:
            logging.warning("Save file path not provided for single channel pair visualization.")
        plt.close()
    
def visualize_single_channel_pair(bispectrum_results,
                                  seg_metadata,
                                  property_names,
                                  save_file_path=None,
                                  smooth_per_sec=True):
    seizure_events = get_seizure_events(seg_metadata)
    channel_pairs_from_seizure_events = get_channel_pairs_from_seizure_events(seizure_events)
    pair_colors = get_pair_colors(channel_pairs_from_seizure_events)
    
    for channel_pair in channel_pairs_from_seizure_events:
        seizure_event_time_windows = get_seizure_event_time_windows_for_a_channel_pair(seizure_events, channel_pair)
        pair_color = pair_colors[channel_pair]
        
        do_visualize_single_channel_pair(
            bispectrum_results=bispectrum_results,
            channel_pair=channel_pair,
            property_names=property_names,
            seizure_event_time_windows=seizure_event_time_windows,
            pair_color=pair_color,
            save_file_path=save_file_path,
            smooth_per_sec=smooth_per_sec
        )
        
def run_aggregate_visualization(aggregated_stats_file_path,
                                patient_name, 
                                visualization_save_path, 
                                visualize_property_names):
    '''Heatmap matrix'''
    logger.info("Running aggregate visualization, plotting heatmap matrix...")
    
    n_channels = len(STANDARD_EEG_CHANNELS)
    
    # results[seizure/non_seizure][channel_pair][band_pair][property_name] = value
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for property_name in visualize_property_names:
        combined_df = []

        for root, _, files in os.walk(aggregated_stats_file_path):
            for file in files:
                if file == 'aggregated_statistics.csv' and os.path.basename(root) == property_name:
                    file_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(file_path)
                        combined_df.append(df)
                    except Exception as e:
                        logger.error(f"Failed to read aggregated stats file {file_path}: {e}")
        if not combined_df:
            logger.warning(f"No valid aggregated stats files found for property {property_name}; skipping visualization.")
            continue
        df = pd.concat(combined_df, ignore_index=True)                
        
        try:
            band_diff_matrices = {band_pair: np.zeros((n_channels, n_channels)) for band_pair in BAND_PAIRS}
            for ch_idx1 in range(n_channels):
                for ch_idx2 in range(ch_idx1 + 1, n_channels):
                    band_pair_stats = df[df['Channel_Pair'] == str((ch_idx1, ch_idx2))]
                    for band_pair in BAND_PAIRS:
                        band_data = band_pair_stats[band_pair_stats['Frequency_Band_Pair'] == str(band_pair)]
                        if not band_data.empty:
                            seizure_means = band_data[f'Seizure_Mean_{property_name.title()}'].values
                            total_seizure_points = band_data['Total_N_Non_Seizure'].values
                            
                            non_seizure_means = band_data[f'Non_Seizure_Mean_{property_name.title()}'].values
                            total_non_seizure_points = band_data['Total_N_Non_Seizure'].values
                            
                            weighted_seizure_mean = np.average(seizure_means, weights=total_seizure_points)
                            weighted_non_seizure_mean = np.average(non_seizure_means, weights=total_non_seizure_points)
                            
                            results['seizure'][(ch_idx1, ch_idx2)][band_pair][property_name] = weighted_seizure_mean
                            results['non_seizure'][(ch_idx1, ch_idx2)][band_pair][property_name] = weighted_non_seizure_mean                            
                            
                            diff = weighted_seizure_mean - weighted_non_seizure_mean
                            
                            band_diff_matrices[band_pair][ch_idx1, ch_idx2] = diff
                            band_diff_matrices[band_pair][ch_idx2, ch_idx1] = diff
            _, axes = plt.subplots(5, 5, figsize=(25, 25))
            axes = axes.flatten()
            for band_idx1, band_name1 in enumerate(BAND_NAMES):
                for band_idx2, band_name2 in enumerate(BAND_NAMES):
                    band_pair = (band_name1, band_name2)
                    ax_idx = band_idx1 * 5 + band_idx2
                    ax = axes[ax_idx]
                    band_pair = band_pair if band_pair in band_diff_matrices else (band_name2, band_name1)
                    diff_matrix = band_diff_matrices[band_pair]
                    vmax = max(abs(np.nanmax(diff_matrix)), abs(np.nanmin(diff_matrix)))
                    vmin = -vmax
                    sns.heatmap(diff_matrix, ax=ax, cmap='seismic', center=0, vmin=vmin, vmax=vmax,
                                xticklabels=STANDARD_EEG_CHANNELS, yticklabels=STANDARD_EEG_CHANNELS, square=True,
                                cbar_kws={'label': f'Seizure - Non-seizure {property_name.title()} Diff'})
                    ax.set_title(f'{band_pair[0]} - {band_pair[1]}')
            plt.suptitle(f'Aggregated {property_name.title()} Difference Between Seizure and Non-Seizure Periods (5x5 Band Pairs)', fontsize=16, y=1)
            plt.tight_layout()
            
            output_file_path = os.path.join(visualization_save_path, patient_name,
                                            f'aggregated_heatmap_{property_name}_seizure_non_seizure_difference.png') if visualization_save_path else None
            if output_file_path:
                # output_file_path's directory name 
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
                plt.close()
                logging.info(f"Aggregated heatmap figure saved to {output_file_path}")
                
        except Exception as e:
            logger.error(f"Failed to generate aggregated heatmap for {property_name}: {e}")
            
    return results


