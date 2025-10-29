import os
import logging
import concurrent.futures
from my_pipeline.data_loader import readers
from my_pipeline.metadata import calculator
from my_pipeline.processing import segment_workflow, stat, network_analysis, visualization
from my_pipeline.persistence import writers


logger = logging.getLogger(__name__)

def run(cfg):
    logger.info("Stage 1: Loading seizure segments meta data from json file...")
    seizure_segments_meta_data = readers.load_segments_from_json(cfg.get('paths', 'segments_json'))
    logger.info(f"Loaded {len(seizure_segments_meta_data)} seizure segments from metadata.")
    
    logger.info("Stage 2: Selecting segments meta data ...")
    if cfg.getboolean('data_loader', 'load_by_patient'):
        patients = [p.strip() 
                    for p in cfg.get('data_loader', 'patients').split(',') 
                    if p.strip()]
        if not patients:
            logger.error("No patients specified in configuration for loading by patient.")
            return
        
        seizure_segments_meta_data = calculator.select_data_by_patient(seizure_segments_meta_data, patients)
        
    elif cfg.getboolean('data_loader', 'load_by_segment'):
        segments = [s.strip() 
                    for s in cfg.get('data_loader', 'segments').split(',') 
                    if s.strip()]
        if not segments:
            logger.error("No segments specified in configuration for loading by segment.")
            return
        
        seizure_segments_meta_data = calculator.select_data_by_segment(seizure_segments_meta_data, segments)
        
    else: # Be careful with this default behavior
        logger.info("No specific loading criteria provided; loading all segments.")
        
    logger.info(f"After selection, {len(seizure_segments_meta_data)} segments remain for processing.")
    
    # group seizure_segments_meta_data by patient for aggregation
    patient_to_segments = {}
    for seg_meta in seizure_segments_meta_data:
        patient_to_segments.setdefault(seg_meta.patient, []).append(seg_meta)
    
    max_workers = cfg.getint('pipeline_modules', 'max_workers', fallback=os.cpu_count()-1)
    logger.info("Stage 3: Processing segments in parallel using up to %d workers...", max_workers)

    for segments in patient_to_segments.values():
        logger.info(f"Patient '{segments[0].patient}' has {len(segments)} segments to process.")
        segment_results = []
    
        if max_workers > 1: #TODO implement parallel processing
            logger.info(f"Using {max_workers} workers for parallel processing. TBD")
            return
        else:
            logger.info("Max workers set to 1; processing segments sequentially in the main process.")
            train_path_h5, tmp_path = cfg.get('paths', 'train_path_h5'), cfg.get('paths', 'tmp_path')
            
            for seg in segments:
                run_cwt = cfg.getboolean('pipeline_modules', 'run_cwt')
                
                run_bispectrum = cfg.getboolean('pipeline_modules', 'run_bispectrum')
                bispectrum_smoothed = cfg.getboolean('run_bispectrum_params', 'bispectrum_smoothed', fallback=True)
                
                run_visualization = cfg.getboolean('pipeline_modules', 'run_visualization')
                
                # Get visualization parameters
                visualization_property_names = cfg.get('run_visualization_params', 'property_names')
                if visualization_property_names:
                    visualization_property_names = [name.strip() for name in visualization_property_names.split(',') if name.strip()]
                else:
                    logger.error("No visualization property names specified in configuration.")
                    visualization_property_names = []
                    
                visualization_smooth_per_sec = cfg.getboolean('run_visualization_params', 'smooth_per_sec', fallback=True)
                visualization_save_path = cfg.get('run_visualization_params', 'save_path', fallback='results/visualizations/')
                visualization_all_channels = cfg.getboolean('run_visualization_params', 'all_channels', fallback=True)
                visualization_single_channel = cfg.getboolean('run_visualization_params', 'single_channel', fallback=False)
                
                run_stat = cfg.getboolean('pipeline_modules', 'run_stat')
                stat_save_path = cfg.get('run_stat_params', 'save_path', fallback='results/stats/')
                
                try:
                    segment_result = segment_workflow.run(train_path_h5, tmp_path, seg, 
                                            run_cwt, run_bispectrum, 
                                            bispectrum_smoothed,  
                                            run_visualization,
                                            visualization_property_names,
                                            visualization_smooth_per_sec,
                                            visualization_save_path,
                                            visualization_all_channels,
                                            visualization_single_channel,
                                            run_stat,
                                            stat_save_path)
                    segment_results.append(segment_result)
                    
                    logger.info(f"Segment '{seg.segment}' processed successfully.")
                except Exception as e:
                    logger.error(f"Segment '{seg.segment}' processing failed: {e}")
        

        # Aggregate results
        aggregate_results = None
        run_aggregate_stat = cfg.getboolean('pipeline_modules', 'run_aggregate_stat', fallback=False)
        # Get aggregate parameters
        if run_aggregate_stat:
            logger.info("Stage 4: Aggregating segment results...")
            aggregate_segment_stats = cfg.getboolean('run_aggregate_stat_params', 'aggregate_segment_stats', fallback=True)
            aggregate_segment_results = cfg.getboolean('run_aggregate_stat_params', 'aggregate_segment_results', fallback=True)
            if aggregate_segment_stats:
                logger.info("Aggregating segment statistics...")
                aggregate_results = stat.run_aggregate_stat(segments, 
                                                            stat_save_path,
                                                            visualization_property_names)
                stat.run_aggregate_global_stat(segments, 
                                            stat_save_path, 
                                            visualization_property_names) # For visualization purpose
            elif aggregate_segment_results:
                logger.info("Aggregating segment results...")
                aggregate_results = stat.run_aggregate_results(segment_results, segments, 
                                                            stat_save_path, visualization_property_names)
                stat.run_aggregate_global_results(segment_results,
                                            segments, 
                                            stat_save_path, visualization_property_names) # For visualization purpose
            else:
                logger.info("No aggregation option selected; skipping aggregation step.")
                
            run_aggregate_visualization = cfg.getboolean('pipeline_modules', 'run_aggregate_visualization', fallback=True)
            aggregated_stats_file_path = cfg.get('run_aggregate_visualization_params', 'aggregated_stats_file_path', fallback=None)
            if run_aggregate_visualization:
                logger.info("Stage 5: Running visualization aggregation...")
                aggregate_results = visualization.run_aggregate_visualization(aggregated_stats_file_path, 
                                                                                segments[0].patient,
                                                                                visualization_save_path,
                                                                                visualization_property_names)
        
        # Run network analysis aggregation
        run_network_analysis = cfg.getboolean('pipeline_modules', 'run_network_analysis', fallback=False)
        if run_network_analysis:
            logger.info("Stage 5: Running network analysis aggregation...")
            network_analysis.run(aggregate_results, segments, visualization_save_path, visualization_property_names)