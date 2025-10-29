import logging
import os
import json
logger = logging.getLogger(__name__)

def save_results(results_dict, output_dir):
    """
    Saves the results from the pipeline to disk.
    """
    logger.info(f"Persisting results to {output_dir}...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历并行结果字典
    for module_name, result_data in results_dict.items():
        filename = os.path.join(output_dir, f"{module_name}_result.json")
        try:
            with open(filename, 'w') as f:
                # 将结果保存为 JSON
                json.dump({"result": result_data}, f, indent=4)
            logger.info(f"Saved result for {module_name} to {filename}")
        except Exception as e:
            logger.error(f"Failed to save result for {module_name}: {e}")
