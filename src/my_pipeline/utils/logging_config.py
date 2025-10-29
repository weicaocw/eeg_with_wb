import logging
import sys

def setup_logging(cfg):
    """
    Configures logging based on the config file.
    """
    log_level = cfg.get('logging', 'log_level', fallback='INFO')
    log_file = cfg.get('logging', 'log_file', fallback='pipeline.log')

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)-8s] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging setup complete.")
