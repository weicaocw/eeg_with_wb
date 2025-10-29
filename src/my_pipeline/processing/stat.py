import logging  
import os
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy import stats
from my_pipeline.processing.utils import (get_seizure_events,
                                          get_seizure_event_time_windows,
                                          get_seizure_event_time_windows_for_a_channel_pair,
                                          get_channel_pairs_from_seizure_events,
                                          create_seizure_mask)

logger = logging.getLogger(__name__)


def calculate_stat_significance_for_a_single_ch_pair(data_by_band_pair, channel_pair,
                                                     mask, property_name, for_aggregate_only=False):
    result_for_all_band_pairs = []
    seizure_mask = mask
    non_seizure_mask = ~mask
    n_seizure = np.sum(seizure_mask)
    n_non_seizure = np.sum(non_seizure_mask)

    for band_pair, _ in data_by_band_pair.items():
        data = data_by_band_pair[band_pair][property_name]
        seizure_values = data[seizure_mask]
        non_seizure_values = data[non_seizure_mask]
        seizure_mean = np.mean(seizure_values)
        seizure_std = np.std(seizure_values)
        non_seizure_mean = np.mean(non_seizure_values)
        non_seizure_std = np.std(non_seizure_values)
        
        statistic, p_value = stats.mannwhitneyu(seizure_values, non_seizure_values, alternative='two-sided')
        t_statistic, t_p_value = stats.ttest_ind(seizure_values, non_seizure_values)
        pooled_std = np.sqrt(((n_seizure - 1) * seizure_std**2 + (n_non_seizure - 1) * non_seizure_std**2) / (n_seizure + n_non_seizure - 2))
        cohens_d = (seizure_mean - non_seizure_mean) / pooled_std if pooled_std > 0 else 0
        
        
        result = {
            'Channel_Pair': channel_pair,
            'Frequency_Band_Pair': band_pair,
            f'Seizure_Mean_{property_name.title()}': seizure_mean,
            'Seizure_Std': seizure_std,
            f'Non_Seizure_Mean_{property_name.title()}': non_seizure_mean,
            'Non_Seizure_Std': non_seizure_std,
            'Diff_Absolute': seizure_mean - non_seizure_mean,
            'Diff_Percent': ((seizure_mean - non_seizure_mean) / non_seizure_mean * 100) if non_seizure_mean != 0 else 0,
            'N_Seizure_Points': n_seizure,
            'N_Non_Seizure_Points': n_non_seizure,
            'Mann_Whitney_U': statistic,
            'P_Value_MW': p_value,
            'T_Statistic': t_statistic,
            'P_Value_TTest': t_p_value,
            'Cohens_D': cohens_d,
            'Significant_MW_005': 'Yes' if p_value < 0.05 else 'No',
            'Significant_MW_001': 'Yes' if p_value < 0.01 else 'No'}
        if for_aggregate_only:
            if for_aggregate_only:
                seizure_mean_col = f'Seizure_Mean_{property_name.title()}'
                non_seizure_mean_col = f'Non_Seizure_Mean_{property_name.title()}'
                required_cols = ['Channel_Pair', 'Frequency_Band_Pair', 'P_Value_MW', 'P_Value_TTest', 'Cohens_D', 'N_Seizure_Points', 'N_Non_Seizure_Points',
                                seizure_mean_col, non_seizure_mean_col]
                # Filter result to only include required columns
                result = {key: value for key, value in result.items() if key in required_cols}
        result_for_all_band_pairs.append(result)
    return result_for_all_band_pairs
        
def calculate_stat_significance(bispectrum_results, masks, property_name):
    results_list = []
    
    for pair, mask in masks.items():
        channel_pair = tuple(pair)
        if channel_pair not in bispectrum_results:
            channel_pair = (channel_pair[1], channel_pair[0])
            if channel_pair not in bispectrum_results:
                logging.warning(f"{channel_pair} not found in bispectrum_results")
                continue
        
        data_by_band_pair = bispectrum_results[channel_pair]
        stats_result = calculate_stat_significance_for_a_single_ch_pair(
            data_by_band_pair, channel_pair, mask, property_name)
        results_list.extend(stats_result)
    return results_list

def run(data, seg_metadata, property_names, save_path):
    logger.info(f"Doing statistical processing for segment {seg_metadata.segment}.")
    
    # Create masks for seizure and non-seizure periods
    seizure_events = get_seizure_events(seg_metadata)
    
    channel_pairs_involved = get_channel_pairs_from_seizure_events(seizure_events)
    seizure_masks = {}
    for pair in channel_pairs_involved:
        seizure_mask = create_seizure_mask(
            get_seizure_event_time_windows_for_a_channel_pair(seizure_events, pair), 
            seg_metadata.data_points)
        seizure_masks[pair] = seizure_mask
    
    # Calculate statistical significance for each requested property
    all_stats_results = {}
    for property_name in property_names:
        stats_results = calculate_stat_significance(data, seizure_masks, property_name)
        all_stats_results[property_name] = stats_results
    
    # Save results to CSV files
    if save_path:
        save_path = os.path.join(save_path, seg_metadata.patient)
        for property_name, stats_results in all_stats_results.items():
            df = pd.DataFrame(stats_results)
            # use property name as a sub directory
            property_save_path = os.path.join(save_path, property_name)
            os.makedirs(property_save_path, exist_ok=True)
            save_file = os.path.join(property_save_path, f'statistics_{seg_metadata.segment}.csv')
            df.to_csv(save_file, index=False)
            logger.info(f"Statistical results for property '{property_name}' saved to {save_file}.")
    return all_stats_results
  
def run_global(data, seg_metadata, property_names, save_path, for_aggregate_only=True):
    logger.info(f"Doing statistical processing for segment {seg_metadata.segment}.")
    
    # Create masks for seizure and non-seizure periods
    seizure_events = get_seizure_events(seg_metadata)
    global_seizure_windows = get_seizure_event_time_windows(seizure_events)
    global_seizure_mask = create_seizure_mask(global_seizure_windows, seg_metadata.data_points)
    
    # Calculate statistical significance for each requested property
    all_stats_results = {}
    for property_name in property_names:
        for channel_pair, data_by_band_pair in data.items():  
            stats_result = calculate_stat_significance_for_a_single_ch_pair(
                data_by_band_pair, channel_pair, 
                global_seizure_mask, property_name, 
                for_aggregate_only=for_aggregate_only)
            if property_name not in all_stats_results:
                all_stats_results[property_name] = []
            # merge the list of results
            all_stats_results[property_name].extend(stats_result)
    
    # Save results to CSV files
    if save_path:
        save_path = os.path.join(save_path, "global", seg_metadata.patient)
        for property_name, stats_results in all_stats_results.items():
            df = pd.DataFrame(stats_results)
            # use property name as a sub directory
            property_save_path = os.path.join(save_path, property_name)
            os.makedirs(property_save_path, exist_ok=True)
            save_file = os.path.join(property_save_path, f'statistics_{seg_metadata.segment}.csv')
            df.to_csv(save_file, index=False)
            logger.info(f"Statistical results for property '{property_name}' saved to {save_file}.")
    return all_stats_results
  
def combine_p_values_fisher(p_values):
    p_values = np.array(p_values)
    p_values = p_values[~np.isnan(p_values) & (p_values > 0) & (p_values <= 1)]
    if len(p_values) == 0: return np.nan
    if len(p_values) == 1: return p_values[0]
    epsilon = 1e-16
    chi2_stat = -2 * np.sum(np.log(np.maximum(p_values, epsilon)))
    df = 2 * len(p_values)
    combined_p = stats.chi2.sf(chi2_stat, df)
    return combined_p

def combine_cohens_d_weighted(d_values, n1_values, n2_values):
    d_values = np.array(d_values)
    n1_values = np.array(n1_values)
    n2_values = np.array(n2_values)
    valid_indices = ~np.isnan(d_values) & (n1_values > 1) & (n2_values > 1) 
    d_values = d_values[valid_indices]
    n1_values = n1_values[valid_indices]
    n2_values = n2_values[valid_indices]
    if len(d_values) == 0: return np.nan
    if len(d_values) == 1: return d_values[0]
    var_d = (n1_values + n2_values) / (n1_values * n2_values) + \
            np.square(d_values) / (2 * (n1_values + n2_values))
    epsilon = 1e-16
    weights = 1.0 / (var_d + epsilon)
    weighted_mean_d = np.sum(weights * d_values) / np.sum(weights)
    return weighted_mean_d

def run_aggregate_stat(seizure_segments_meta_data, save_path, property_names):
    logger.info("Aggregating statistical results across all segments.")
    
    if not save_path:
        logger.error("No save path provided for aggregated statistics.")
        return

    all_dfs = defaultdict(list)
    for property_name in property_names:
        for seg_metadata in seizure_segments_meta_data:
            segment_id = seg_metadata.segment
            patient_id = seg_metadata.patient
            stats_file = os.path.join(save_path, patient_id, property_name, f'statistics_{segment_id}.csv')
            if not os.path.isfile(stats_file):
                logger.warning(f"Statistics file {stats_file} not found; skipping.")
                continue
            try:
                df = pd.read_csv(stats_file)
                all_dfs[property_name].append(df)
            except pd.errors.EmptyDataError:
                logger.error(f"Statistics file {stats_file} is empty; skipping.")
                continue
            except Exception as e:
                logger.error(f"Error reading {stats_file}: {e}")
                continue
    if not all_dfs:
        logger.error("No statistical data found to aggregate.")
        return
    
    aggregated_stats = {}
    for property_name, df_list in all_dfs.items():
        if not df_list:
            logger.warning(f"No dataframes found for property '{property_name}'; skipping.")
            continue
        combined_df = pd.concat(df_list, ignore_index=True)
    
        try:
            grouped = combined_df.groupby(['Channel_Pair', 'Frequency_Band_Pair'])
        except KeyError as e:
            logger.error(f"Expected columns not found in combined dataframe: {e}")
            return
    
        meta_results = []
        seizure_mean_col = f'Seizure_Mean_{property_name.title()}'
        non_seizure_mean_col = f'Non_Seizure_Mean_{property_name.title()}'
    
        logger.info(f"Calculating meta-statistics for {len(grouped)} groups.")

        for name, group in grouped:
            channel_pair, band_pair = name
            
            # Ensure required columns exist
            required_cols = ['P_Value_MW', 'P_Value_TTest', 'Cohens_D', 
                            'N_Seizure_Points', 'N_Non_Seizure_Points',
                            seizure_mean_col, non_seizure_mean_col]
            if not all(col in group.columns for col in required_cols):
                logging.error(f"Missing required columns in group for channel pair {channel_pair}, band pair {band_pair}. Skipping.")
                continue

            # Combine p-values using Fisher's method
            combined_p_mw = combine_p_values_fisher(group['P_Value_MW'].tolist())
            combined_p_ttest = combine_p_values_fisher(group['P_Value_TTest'].tolist())
            
            # Combine Cohen's d using weighted average
            combined_d = combine_cohens_d_weighted(
                group['Cohens_D'].tolist(),
                group['N_Seizure_Points'].tolist(),
                group['N_Non_Seizure_Points'].tolist()
            )
            
            # Calculate average means
            avg_seizure_mean = group[seizure_mean_col].mean()
            avg_non_seizure_mean = group[non_seizure_mean_col].mean()
            total_n_seizure = group['N_Seizure_Points'].sum()
            total_n_non_seizure = group['N_Non_Seizure_Points'].sum()
            num_segments = len(group) 

            meta_results.append({
                'Channel_Pair': channel_pair,
                'Frequency_Band_Pair': band_pair,
                'Num_Segments': num_segments,
                f'Seizure_Mean_{property_name.title()}': avg_seizure_mean,
                f'Non_Seizure_Mean_{property_name.title()}': avg_non_seizure_mean,
                'Total_N_Seizure': total_n_seizure,
                'Total_N_Non_Seizure': total_n_non_seizure,
                'P_Value_MW': combined_p_mw,
                'P_Value_TTest': combined_p_ttest,
                'Cohens_D': combined_d,
                'Significant_MW_005': 'Yes' if not np.isnan(combined_p_mw) and combined_p_mw < 0.05 else 'No',
                'Significant_MW_001': 'Yes' if not np.isnan(combined_p_mw) and combined_p_mw < 0.01 else 'No'
            })
      
        aggregated_stats[property_name] = meta_results
          
    logging.info("Meta-statistics calculation completed.")
    
    # Save aggregated results to CSV files
    if save_path:
        #TODO Generally we aggregate by patient
        save_path = os.path.join(save_path, seizure_segments_meta_data[0].patient)
        for property_name, stats_results in aggregated_stats.items():
            df = pd.DataFrame(stats_results)
            property_save_path = os.path.join(save_path, property_name)
            os.makedirs(property_save_path, exist_ok=True)
            save_file = os.path.join(property_save_path, 
                                     f'aggregated_statistics.csv')
            df.to_csv(save_file, index=False)
            logger.info(f"Aggregated statistical results for property '{property_name}' saved to {save_file}.")
    
    return aggregated_stats
        
def run_aggregate_results(segment_results, seizure_segments_meta_data, save_path, property_names):
    logger.info("Aggregating across all segments for statistical results.")
    
    if len(segment_results) != len(seizure_segments_meta_data):
        logger.error("Mismatch between number of segment results and segment metadata.")
        return
    
    total_seizure_values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    total_non_seizure_values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for bispectrum_results, seg_metadata in zip(segment_results, seizure_segments_meta_data):
        segment_id = seg_metadata.segment
        logger.info(f"Processing aggregation for segment {segment_id}.")
        
        # Create masks for seizure and non-seizure periods
        seizure_events = get_seizure_events(seg_metadata)
        channel_pairs_involved = get_channel_pairs_from_seizure_events(seizure_events)
        seizure_masks = {}
        for pair in channel_pairs_involved:
            seizure_mask = create_seizure_mask(
                get_seizure_event_time_windows_for_a_channel_pair(seizure_events, pair), 
                seg_metadata.data_points)
            seizure_masks[pair] = seizure_mask
        

        for pair, mask in seizure_masks.items():
            channel_pair = tuple(pair)
            if channel_pair not in bispectrum_results:
                channel_pair = (channel_pair[1], channel_pair[0])
                if channel_pair not in bispectrum_results:
                    logging.warning(f"{channel_pair} not found in bispectrum_results")
                    continue
            
            data_by_band_pair = bispectrum_results[channel_pair]

            for band_pair, data in data_by_band_pair.items():
                for property_name in property_names:
                    data = data[property_name]
                    seizure_values = data[mask]
                    non_seizure_values = data[~mask]
                    total_seizure_values[channel_pair][band_pair][property_name].extend(seizure_values)
                    total_non_seizure_values[channel_pair][band_pair][property_name].extend(non_seizure_values)
    
    # Now calculate aggregated statistics
    aggregated_results = {}
    for property_name in property_names:
        stats_results = []
        for channel_pair, band_data in total_seizure_values.items():
            for band_pair, prop_data in band_data.items():
                seizure_values = np.array(prop_data[property_name])
                non_seizure_values = np.array(total_non_seizure_values[channel_pair][band_pair][property_name])
                
                n_seizure = len(seizure_values)
                n_non_seizure = len(non_seizure_values)
                
                if n_seizure == 0 or n_non_seizure == 0:
                    logger.error(f"Insufficient data for channel pair {channel_pair}, band pair {band_pair}. Skipping.")
                    continue
                
                seizure_mean = np.mean(seizure_values)
                seizure_std = np.std(seizure_values)
                non_seizure_mean = np.mean(non_seizure_values)
                non_seizure_std = np.std(non_seizure_values)
                
                statistic, p_value = stats.mannwhitneyu(seizure_values, non_seizure_values, alternative='two-sided')
                t_statistic, t_p_value = stats.ttest_ind(seizure_values, non_seizure_values)
                pooled_std = np.sqrt(((n_seizure - 1) * seizure_std**2 + (n_non_seizure - 1) * non_seizure_std**2) / (n_seizure + n_non_seizure - 2))
                cohens_d = (seizure_mean - non_seizure_mean) / pooled_std if pooled_std > 0 else 0
                
                stats_results.append({
                    'Channel_Pair': channel_pair,
                    'Frequency_Band_Pair': band_pair,
                    f'Seizure_Mean_{property_name.title()}': seizure_mean,
                    'Seizure_Std': seizure_std,
                    f'Non_Seizure_Mean_{property_name.title()}': non_seizure_mean,
                    'Non_Seizure_Std': non_seizure_std,
                    'Diff_Absolute': seizure_mean - non_seizure_mean,
                    'Diff_Percent': ((seizure_mean - non_seizure_mean) / non_seizure_mean * 100) if non_seizure_mean != 0 else 0,
                    'N_Seizure_Points': n_seizure,
                    'N_Non_Seizure_Points': n_non_seizure,
                    'Mann_Whitney_U': statistic,
                    'P_Value_MW': p_value,
                    'T_Statistic': t_statistic,
                    'P_Value_TTest': t_p_value,
                    'Cohens_D': cohens_d,
                    'Significant_MW_005': 'Yes' if p_value < 0.05 else 'No',
                    'Significant_MW_001': 'Yes' if p_value < 0.01 else 'No'})
        aggregated_results[property_name] = stats_results   
        
        
    # Save aggregated results to CSV files
    if save_path:
        #TODO Generally we aggregate by patient
        save_path = os.path.join(save_path, seizure_segments_meta_data[0].patient)
        for property_name, stats_results in aggregated_results.items():
            df = pd.DataFrame(stats_results)
            property_save_path = os.path.join(save_path, property_name)
            os.makedirs(property_save_path, exist_ok=True)
            save_file = os.path.join(property_save_path, 
                                     f'aggregated_statistics.csv')
            df.to_csv(save_file, index=False)
            logger.info(f"Aggregated statistical results for property '{property_name}' saved to {save_file}.")
    
    return aggregated_results

def run_aggregate_global_stat(seizure_segments_meta_data, save_path, property_names):
    logger.info("Aggregating global statistical results across all segments.")
    save_path = os.path.join(save_path, "global")
    return run_aggregate_stat(seizure_segments_meta_data, save_path, property_names)
    
def run_aggregate_global_results(segment_results, seizure_segments_meta_data, save_path, property_names):
    logger.info("Aggregating global results across all segments for statistical results.")
    
    if len(segment_results) != len(seizure_segments_meta_data):
        logger.error("Mismatch between number of segment results and segment metadata.")
        return
    
    total_seizure_values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    total_non_seizure_values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for bispectrum_results, seg_metadata in zip(segment_results, seizure_segments_meta_data):
        segment_id = seg_metadata.segment
        logger.info(f"Processing aggregation for segment {segment_id}.")
        
        # Create global mask for seizure and non-seizure periods
        seizure_events = get_seizure_events(seg_metadata)
        global_seizure_windows = get_seizure_event_time_windows(seizure_events)
        global_seizure_mask = create_seizure_mask(global_seizure_windows, seg_metadata.data_points)
        
        for channel_pair, data_by_band_pair in bispectrum_results.items():
            for band_pair, data in data_by_band_pair.items():
                for property_name in property_names:
                    data = data[property_name]
                    seizure_values = data[global_seizure_mask]
                    non_seizure_values = data[~global_seizure_mask]
                    total_seizure_values[channel_pair][band_pair][property_name].extend(seizure_values)
                    total_non_seizure_values[channel_pair][band_pair][property_name].extend(non_seizure_values)
    
    # Now calculate aggregated statistics
    aggregated_results = {}
    for property_name in property_names:
        stats_results = []
        for channel_pair, band_data in total_seizure_values.items():
            for band_pair, prop_data in band_data.items():
                seizure_values = np.array(prop_data[property_name])
                non_seizure_values = np.array(total_non_seizure_values[channel_pair][band_pair][property_name])
                
                n_seizure = len(seizure_values)
                n_non_seizure = len(non_seizure_values)
                
                if n_seizure == 0 or n_non_seizure == 0:
                    logger.error(f"Insufficient data for channel pair {channel_pair}, band pair {band_pair}. Skipping.")
                    continue
                
                seizure_mean = np.mean(seizure_values)
                seizure_std = np.std(seizure_values)
                non_seizure_mean = np.mean(non_seizure_values)
                non_seizure_std = np.std(non_seizure_values)
                
                statistic, p_value = stats.mannwhitneyu(seizure_values, non_seizure_values, alternative='two-sided')
                t_statistic, t_p_value = stats.ttest_ind(seizure_values, non_seizure_values)
                pooled_std = np.sqrt(((n_seizure - 1) * seizure_std**2 + (n_non_seizure - 1) * non_seizure_std**2) / (n_seizure + n_non_seizure - 2))
                cohens_d = (seizure_mean - non_seizure_mean) / pooled_std if pooled_std > 0 else 0
                
                stats_results.append({
                    'Channel_Pair': channel_pair,
                    'Frequency_Band_Pair': band_pair,
                    f'Seizure_Mean_{property_name.title()}': seizure_mean,
                    'Seizure_Std': seizure_std,
                    f'Non_Seizure_Mean_{property_name.title()}': non_seizure_mean,
                    'Non_Seizure_Std': non_seizure_std,
                    'Diff_Absolute': seizure_mean - non_seizure_mean,
                    'Diff_Percent': ((seizure_mean - non_seizure_mean) / non_seizure_mean * 100) if non_seizure_mean != 0 else 0,
                    'N_Seizure_Points': n_seizure,
                    'N_Non_Seizure_Points': n_non_seizure,
                    'Mann_Whitney_U': statistic,
                    'P_Value_MW': p_value,
                    'T_Statistic': t_statistic,
                    'P_Value_TTest': t_p_value,
                    'Cohens_D': cohens_d,
                    'Significant_MW_005': 'Yes' if p_value < 0.05 else 'No',
                    'Significant_MW_001': 'Yes' if p_value < 0.01 else 'No'})
        aggregated_results[property_name] = stats_results   
        
        
    # Save aggregated results to CSV files
    if save_path:
        #TODO Generally we aggregate by patient
        save_path = os.path.join(save_path, "global", seizure_segments_meta_data[0].patient)
        for property_name, stats_results in aggregated_results.items():
            df = pd.DataFrame(stats_results)
            property_save_path = os.path.join(save_path, property_name)
            os.makedirs(property_save_path, exist_ok=True)
            save_file = os.path.join(property_save_path, 
                                     f'aggregated_statistics.csv')
            df.to_csv(save_file, index=False)
            logger.info(f"Aggregated statistical results for property '{property_name}' saved to {save_file}.")
    
    return aggregated_results