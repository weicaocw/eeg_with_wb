import logging
import numpy as np
from collections import defaultdict
from my_pipeline.processing.constants import (CMAP, STANDARD_EEG_CHANNELS, 
                                              EVENT_LABEL_BACKGROUND,
                                              SAMPLING_RATE)

logger = logging.getLogger(__name__)

def smooth_per_second(data, sfreq):
    n_points = data.shape[-1]
    n_seconds = int(n_points // sfreq)
    smoothed = np.array([
        np.mean(data[..., int(i*sfreq):int((i+1)*sfreq)], axis=-1)
        for i in range(n_seconds)
    ])
    return smoothed.T


def get_seizure_events(seg_metadata):
    return [event for event in seg_metadata.events if event['label'] != EVENT_LABEL_BACKGROUND]


def get_seizure_event_time_windows_for_a_channel_pair(seizure_events, channel_pair):
    ch_name1 = STANDARD_EEG_CHANNELS[channel_pair[0]]
    ch_name2 = STANDARD_EEG_CHANNELS[channel_pair[1]]
    
    relevant_events = [event for event in seizure_events if ch_name1 in event['pair'] and ch_name2 in event['pair']]
    
    if not relevant_events:
        return [(0, 0)]
    
    # Merge time windows
    # Sort events by start time
    relevant_events.sort(key=lambda x: x['start_time'])
    merged_windows = []
    current_start, current_stop = relevant_events[0]['start_time'], relevant_events[0]['stop_time']
    for event in relevant_events[1:]:
        if event['start_time'] <= current_stop:  # Overlap
            current_stop = max(current_stop, event['stop_time'])
        else:
            merged_windows.append((current_start, current_stop))
            current_start, current_stop = event['start_time'], event['stop_time']
    merged_windows.append((current_start, current_stop))
    return merged_windows
    
def get_seizure_event_time_windows(seizure_events):
    if not seizure_events:
        return [(0, 0)]
    
    # Merge time windows
    # Sort events by start time
    seizure_events.sort(key=lambda x: x['start_time'])
    merged_windows = []
    current_start, current_stop = seizure_events[0]['start_time'], seizure_events[0]['stop_time']
    for event in seizure_events[1:]:
        if event['start_time'] <= current_stop:  # Overlap
            current_stop = max(current_stop, event['stop_time'])
        else:
            merged_windows.append((current_start, current_stop))
            current_start, current_stop = event['start_time'], event['stop_time']
    merged_windows.append((current_start, current_stop))
    return merged_windows

def get_channel_pairs_from_seizure_events(seizure_events):
    channel_pairs = set()
    
    for event in seizure_events:
        pair = tuple(event['pair'])
        if len(pair) != 2:
            continue
        
        ch_name1, ch_name2 = pair
        try:
            channel_pair = (STANDARD_EEG_CHANNELS.index(ch_name1) , 
                            STANDARD_EEG_CHANNELS.index(ch_name2))
        except ValueError as e:
            continue
        
        channel_pairs.add(channel_pair)
    
    return channel_pairs
        

def get_pair_colors(channel_pairs):
    return {pair: CMAP(idx % CMAP.N) for idx, pair in enumerate(channel_pairs)}

def create_seizure_mask(seizure_event_time_windows, total_data_points):
    mask = np.zeros(total_data_points, dtype=bool)
    for start_time, stop_time in seizure_event_time_windows:
        start_idx = int(start_time * SAMPLING_RATE)
        stop_idx = int(stop_time * SAMPLING_RATE)
        mask[start_idx:stop_idx] = True
    return mask