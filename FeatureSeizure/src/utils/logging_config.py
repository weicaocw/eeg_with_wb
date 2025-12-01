import logging
import sys
import os
import datetime

def setup_logging(config):
    """
    Configures logging based on the config dict.
    """
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    
    base_log_file = log_config.get('file', 'logs/app.log')

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    root, ext = os.path.splitext(base_log_file)
    log_file = f"{root}_{ts}{ext}"
    log_format = log_config.get('format', '%(asctime)s [%(levelname)-8s] [%(name)s] - %(message)s')
    datefmt = log_config.get('datefmt', '%Y-%m-%d %H:%M:%S')

    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Convert string level to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    stream_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=datefmt,
        handlers=[file_handler, stream_handler],
        force=True
    )
    logging.captureWarnings(True)
    logging.info("Logging setup complete.")