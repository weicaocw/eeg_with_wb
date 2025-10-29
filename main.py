import argparse
import logging
import os
from my_pipeline import config
from my_pipeline.utils import logging_config
from my_pipeline import pipeline

def main():
    parser = argparse.ArgumentParser(description="Run the analysis pipeline.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/default.conf", 
        help="Path to configuration file."
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs='*',  # Allow multiple overrides
        help="Override config values, format: section.key=value (e.g., paths.data_input_file=/new/path)"
    )
    args = parser.parse_args()

    # Load configuration
    cfg = config.load_config(args.config)
    
    # 2. Setup logging
    log_dir = os.path.dirname(cfg.get('logging', 'log_file', fallback='logs/pipeline.log'))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging_config.setup_logging(cfg)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting pipeline with config: {args.config}")
    
        # Apply overrides
    if args.override:
        for override in args.override:
            try:
                section, key_value = override.split('.', 1)
                key, value = key_value.split('=', 1)
                cfg.set(section, key, value)
                logger.info(f"Overridden config: [{section}] {key} = {value}")
            except ValueError:
                logger.error(f"Invalid override format: {override}. Expected format: section.key=value")
    
    # 3. Run pipeline
    try:
        pipeline.run(cfg)
        logger.info("Pipeline finished successfully.")
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
