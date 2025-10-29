import logging
logger = logging.getLogger(__name__)

def compute_metadata(data):
    """
    Mock function to compute metadata.
    """
    logger.info("Mock compute: Calculating metadata...")
    try:
        row_count = len(data.get("col1", []))
    except Exception:
        row_count = 0
        
    metadata = {"row_count": row_count, "source_type": "mock_csv"}
    logger.info(f"Mock compute: Metadata calculated: {metadata}")
    return metadata

def select_data_by_patient(seizure_segments_meta_data, patients):
    """
    Select seizure segments by patient IDs.
    """
    logger.info(f"Selecting segments for patients: {patients}")
    selected_segments = [seg for seg in seizure_segments_meta_data if seg.patient in patients]
    logger.info(f"Selected {len(selected_segments)} segments for specified patients.")
    return selected_segments

def select_data_by_segment(seizure_segments_meta_data, segments):
    """
    Select seizure segments by segment IDs.
    """
    logger.info(f"Selecting segments: {segments}")
    selected_segments = [seg for seg in seizure_segments_meta_data if seg.segment in segments]
    logger.info(f"Selected {len(selected_segments)} segments for specified segment IDs.")
    return selected_segments
