import logging  
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def run(data, segment_id):
    logger.info(f"Normalizing data for segment {segment_id} using StandardScaler.")
    # Transpose to (samples, features) format for StandardScaler
    normalized_data = None
    scaler = StandardScaler()
    try:
        normalized_data = scaler.fit_transform(data.T).T
    except Exception as e:
        logger.error(f"Normalization error for segment {segment_id}: {e}")
        raise

    return normalized_data
