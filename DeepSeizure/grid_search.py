import itertools
import yaml
import copy
from main import run_pipeline
from src.utils import load_config

def grid_search():
    # 1. 定义搜索空间
    search_space = {
        'learning_rate': [1e-3, 1e-4],
        'batch_size': [16, 32],
        'model_type': ['lstm', 'transformer']
    }
    
    # 生成所有组合
    keys, values = zip(*search_space.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Total combinations to search: {len(combinations)}")
    
    # 2. 加载基础配置
    base_config_path = "DeepSeizure/configs/config.yaml"
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
        
    # 3. 循环训练
    for i, params in enumerate(combinations):
        print(f"\n\n=== Grid Search Run {i+1}/{len(combinations)}: {params} ===")
        
        # 深度拷贝并修改配置
        current_config = copy.deepcopy(base_config)
        current_config['train']['learning_rate'] = params['learning_rate']
        current_config['train']['batch_size'] = params['batch_size']
        current_config['train']['model_type'] = params['model_type']
        
        # 保存为临时 Config 文件 (或者修改 main 直接接受 dict)
        # 这里我们修改 main.py 的 run_pipeline 稍微改一下最好，
        # 但为了简单，我们保存一个临时 yaml
        tmp_config_path = f"DeepSeizure/configs/tmp_search_{i}.yaml"
        with open(tmp_config_path, 'w') as f:
            yaml.dump(current_config, f)
            
        # 运行 Pipeline
        try:
            run_pipeline(tmp_config_path)
        except Exception as e:
            print(f"Run failed: {e}")
            
        # 清理
        import os
        if os.path.exists(tmp_config_path):
            os.remove(tmp_config_path)

if __name__ == "__main__":
    grid_search()