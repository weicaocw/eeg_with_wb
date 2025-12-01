import os
import yaml
import numpy as np
import logging
import argparse
from sklearn.model_selection import train_test_split
from src.utils.parser import parse_chb_summary
from src.utils.logging_config import setup_logging
from src.data.loader import CHBDataLoader
from src.models.engine import ModelEngine

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="EEG FeatureSeizure Pipeline")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()

    # 1. 加载配置
    config = load_config(args.config)

    # 2. 设置日志
    setup_logging(config)
    import pprint
    logging.info("Config file content:\n" + pprint.pformat(config))
    logging.info(f"Project: {config['project_name']}")
    summary_path = os.path.join(config['paths']['data_dir'], config['paths']['summary_file'])
    logging.info(f"Parsing summary: {summary_path}")
    seizure_info = parse_chb_summary(summary_path)

    # 3. 初始化数据加载器
    loader = CHBDataLoader(config, seizure_info)

    # 4. 遍历文件构建数据集
    all_X = []
    all_y = []

    data_dir = config['paths']['data_dir']
    edf_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.edf')])

    logging.info("Starting data extraction pipeline...")
    for f_name in edf_files:
        X_f, y_f = loader.process_file(f_name)
        if X_f is not None:
            all_X.append(X_f)
            all_y.append(y_f)

    if not all_X:
        logging.error("No data extracted. Exiting.")
        return

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    logging.info(f"Dataset built. Shape: {X.shape}")
    logging.info(f"Features per sample: {X.shape[1]}")
    # 打印部分特征名称示例
    if loader.feature_names:
        logging.info(f"Feature Names: {loader.feature_names}")

    # 5. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 6. 训练模型
    engine = ModelEngine(config)
    logging.info(f"Training {config['model']['type']}...")
    engine.train(X_train, y_train)

    # 7. 评估
    engine.evaluate(X_test, y_test, feature_names=loader.feature_names)

if __name__ == "__main__":
    main()