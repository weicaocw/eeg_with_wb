import logging
from configparser import ConfigParser

from my_pipeline.data_loader import readers
from my_pipeline.processing import normalization, cwt, bisepctrum, visualization, stat

logger = logging.getLogger(__name__)

def run(train_path_h5, tmp_path, seg_meta, run_cwt, run_bispectrum, 
        bispectrum_smoothed,
        run_visualization, 
        visualization_property_names,
        visualization_smooth_per_sec,
        visualization_save_path,
        visualization_all_channels,
        visualization_single_channel,
        run_stat,
        stat_save_path):
    """
    Run the full processing workflow for a single seizure segment in a separate process.
    This function is intended to be executed in a worker process.
    Parameters:
    """
    
    segment_id = seg_meta.segment
    logger.info(f"[{segment_id}] WORKER: Workflow started in new process.")
    
    try:
        # 1. Load data for the segment
        logger.info(f"[{segment_id}] WORKER: Loading data...")
        data = readers.load_data_from_h5(train_path_h5, tmp_path, seg_meta.file_path)
        
        # 2. Normalize data
        logger.info(f"[{segment_id}] WORKER: Normalizing data...")
        data = normalization.run(data, segment_id)
        
        result = None
        if run_cwt:
            # 3. Run Module 1: Continuous Wavelet Transform (CWT)
            logger.info(f"[{segment_id}] WORKER: Running Module 1: Continuous Wavelet Transform (CWT)...")
            result = cwt.run(data, segment_id) # List of CWT coefficients per channel
        
        if run_bispectrum:
            # 4. Run Module 2: Bispectrum Calculation
            logger.info(f"[{segment_id}] WORKER: Running Module 2: Bispectrum Calculation...")
            result = bisepctrum.run(result, segment_id, bispectrum_smoothed) # Bispectrum results
            
        if run_visualization:
            # 5. Run Module 3: Visualization
            logger.info(f"[{segment_id}] WORKER: Running Module 3: Visualization...")
            if visualization_all_channels:
                visualization.visualize_all_channel_pairs(
                    bispectrum_results=result,
                    seg_metadata=seg_meta,
                    property_names=visualization_property_names,
                    save_file_path=visualization_save_path,
                    smooth_per_sec=visualization_smooth_per_sec
                )
            if visualization_single_channel:
                visualization.visualize_single_channel_pair(
                    bispectrum_results=result,
                    seg_metadata=seg_meta,
                    property_names=visualization_property_names,
                    save_file_path=visualization_save_path,
                    smooth_per_sec=visualization_smooth_per_sec
                )
                
        if run_stat:
            # 6. Run Module 4: Statistical Analysis
            logger.info(f"[{segment_id}] WORKER: Running Module 4: Statistical Analysis...")
            # Placeholder for statistical analysis function
            result, _ = (stat.run(result, seg_metadata=seg_meta, 
                         property_names=visualization_property_names,
                         save_path=stat_save_path), stat.run_global(result, seg_meta, visualization_property_names, 
                                stat_save_path, for_aggregate_only=True)) # For visualization later
        return result
        
    except Exception as e:
        logger.error(f"[{segment_id}] WORKER: An error occurred: {e}", exc_info=True)
        raise