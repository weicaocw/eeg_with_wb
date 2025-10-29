import logging
import os
import json
import shutil
import h5py

logger = logging.getLogger(__name__)

def load_data(filepath):
    """
    Mock function to load data.
    """
    logger.info(f"Mock load: Reading data from {filepath}...")
    # 在真实的应用中, 你会用例如 pd.read_csv(filepath)
    # 为了测试, 我们只返回一个模拟的字典
    mock_data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
    logger.info("Mock load: Data loaded successfully.")
    return mock_data

class SeizureSegment:
    """
    Represents a seizure segment with associated metadata.
    """
    def __init__(self, sfreq, high_pass, low_pass, channels, patient, session, segment, montage, total_duration,
                 seizure_duration, percentage, edf_size_kb, data_points, events):
        self.sfreq = float(sfreq)
        self.high_pass = float(high_pass)
        self.low_pass = float(low_pass)
        self.channels = list(channels)
        
        self.patient = str(patient)
        self.session = str(session)
        self.segment = str(segment)
        self.montage = str(montage)
        self.total_duration_sec = float(total_duration)
        self.seizure_duration_sec = float(seizure_duration)
        self.percentage = float(percentage)
        self.edf_size_kb = float(edf_size_kb)
        self.data_points = int(data_points)
        self.events = list(events)

        self.file_path = os.path.join(self.patient, self.session, self.montage, self.segment)
        
    @classmethod
    def from_dict(cls, data):
        """Reconstruct SeizureSegment from a dictionary."""
        events = [{'pair': frozenset(event['pair']), 'start_time': event['start_time'], 'stop_time': event['stop_time'], 'label': event['label']} for event in data['events']] 
        obj = cls(
            sfreq=data['sfreq'],
            high_pass=data['high_pass'],
            low_pass=data['low_pass'],
            channels=data['channels'],
            patient=data['patient'],
            session=data['session'],
            segment=data['segment'],
            montage=data['montage'],
            total_duration=data['total_duration_sec'],
            seizure_duration=data['seizure_duration_sec'],
            percentage=data['percentage'],
            edf_size_kb=data['edf_size_kb'],
            data_points=data['data_points'],
            events=events
        )
        return obj

    def __repr__(self):
        return (f"<SeizureSegment patient='{self.patient}' "
                f"session='{self.session}' segment='{self.segment}'>")

def load_segments_from_json(file_path):
    """Load seizure segments from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [SeizureSegment.from_dict(item) for item in data]

def load_data_from_h5(train_path_h5, tmp_path, segment_file_path):
    h5_path = os.path.join(train_path_h5, segment_file_path + '.h5')
    tmp_file_path = os.path.join(tmp_path,  os.path.basename(segment_file_path) + '.h5')
    
    logger.info(f"Loading data from H5 file: {h5_path}")
    logger.info(f"Copying to temporary path: {tmp_file_path}")
    shutil.copyfile(h5_path, tmp_file_path)
    logger.info("Data copied to temporary path successfully.")
    
    # Read the data from the temporary H5 file
    try:
        with h5py.File(tmp_file_path, 'r') as f:
            data = f['eeg'][:]
        logger.info(f"Data loaded for {segment_file_path} from temporary H5 file successfully. Shape: {data.shape}")
    except Exception as e:
        logger.error(f"Failed to read data from temporary H5 file: {e}")
        data = None
        
    # Clean up the temporary file
    try:
        os.remove(tmp_file_path)
        logger.info(f"Temporary file {tmp_file_path} removed successfully.")
    except Exception as e:
        logger.error(f"Failed to remove temporary file {tmp_file_path}: {e}")
        
    return data
    
