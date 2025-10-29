import logging
import pywt
from my_pipeline.processing.constants import (
    DEFAULT_CWT_WAVELET, 
    STANDARD_EEG_CHANNELS, 
    SAMPLING_RATE, 
    FREQUENCY_USED)

logger = logging.getLogger(__name__)

def run(data, segment_id):
    """
    Perform Continuous Wavelet Transform (CWT) on the input data.
    """
    logger.info("Running Continuous Wavelet Transform (CWT) for segment %s", segment_id)
    result = {}
    scales = pywt.frequency2scale(DEFAULT_CWT_WAVELET, FREQUENCY_USED/SAMPLING_RATE)
    
    assert data.shape[0] == len(STANDARD_EEG_CHANNELS), \
        f"Data channel count {data.shape[0]} does not match expected {len(STANDARD_EEG_CHANNELS)}"
    
    for idx, channel_name in enumerate(STANDARD_EEG_CHANNELS):
        logger.info(f"CWT: Processing channel {channel_name} ({idx+1}/{len(STANDARD_EEG_CHANNELS)}) for segment {segment_id}") 
        signals = data[idx, :]
        
        try:
            coefficients, _ = pywt.cwt(signals, scales, DEFAULT_CWT_WAVELET, sampling_period=1/SAMPLING_RATE)
        except Exception as e:
            logger.error(f"CWT: Error processing channel {channel_name} for segment {segment_id}: {e}")
            raise    
        
        result[idx] = coefficients
        logger.info(f"CWT: Completed channel {channel_name} for segment {segment_id}")
        
    return result
        


