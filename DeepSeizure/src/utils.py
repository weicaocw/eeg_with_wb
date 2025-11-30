import yaml
import argparse

def load_config(config_path=None):
    # 1. 优先读取命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='DeepSeizure/configs/config.yaml', help='Path to config file')
    # 允许命令行覆盖 Ablation 参数
    parser.add_argument('--no_bispectrum', action='store_true', help='Disable bispectrum calculation')
    args, _ = parser.parse_known_args()
    
    path = config_path if config_path else args.config
    
    # 2. 读取 YAML
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        
    # 3. 命令行覆盖
    if args.no_bispectrum:
        config['features']['use_bispectrum'] = False
        print(">> [Config] Overwriting: use_bispectrum = False")
        
    return config