import subprocess
import os
import time
import datetime

# ================= 配置区域 =================

# 1. 定义要运行的 Config 文件列表
# 注意：假设这些文件都在 DeepSeizure/configs/ 目录下
# 如果在其他地方，请修改下面的路径前缀
CONFIG_FILES = [
    # standard ablation configs
    "config_bi_lstm.yaml",
    "config_bi_tran.yaml",
    "config_nobi_lstm.yaml",
    "config_nobi_tran.yaml",

    # raw signal input only (no normalization) configs
    "config_bi_raw_lstm.yaml",
    "config_bi_raw_tran.yaml",    
    
    # no reduction ablation configs    
    "config_bi_nored_lstm.yaml",
    "config_bi_nored_tran.yaml",
    
    # raw only + no reduction configs
    "config_bi_raw_nored_lstm.yaml",
    "config_bi_raw_nored_tran.yaml",
]

# 2. 脚本路径 (相对于当前目录)
MAIN_SCRIPT = "/root/autodl-tmp/code/eeg_with_wb/DeepSeizure/main.py"
CONFIG_DIR = "/root/autodl-tmp/code/eeg_with_wb/DeepSeizure/configs"

# 3. 日志保存目录
LOG_DIR = "/root/autodl-tmp/code/eeg_with_wb/DeepSeizure/batch_logs"

# ===========================================

def run_experiments():
    # 创建日志目录
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        print(f"Created log directory: {LOG_DIR}")

    total = len(CONFIG_FILES)
    print(f"Found {total} experiments to run.\n")

    for i, config_name in enumerate(CONFIG_FILES):
        # 1. 构建完整路径
        config_path = os.path.join(CONFIG_DIR, config_name)
        
        # 2. 构建日志文件名 (包含时间戳，防止覆盖)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 去掉 .yaml 后缀
        exp_name = os.path.splitext(config_name)[0]
        log_file_path = os.path.join(LOG_DIR, f"{exp_name}_{timestamp}.log")
        
        print(f"[{i+1}/{total}] Starting experiment: {config_name}")
        print(f"       Log file: {log_file_path}")
        
        start_time = time.time()

        # 3. 执行命令
        # 命令格式: python DeepSeizure/main.py --config DeepSeizure/configs/xxx.yaml
        cmd = ["python", MAIN_SCRIPT, "--config", config_path]
        
        try:
            # 打开日志文件准备写入
            with open(log_file_path, "w") as log_file:
                # subprocess.run 会阻塞直到命令完成
                # stdout=log_file, stderr=subprocess.STDOUT 把所有输出都重定向到文件
                result = subprocess.run(
                    cmd, 
                    stdout=log_file, 
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            duration = time.time() - start_time
            
            # 4. 检查结果
            if result.returncode == 0:
                print(f"       ✅ Finished successfully in {duration/60:.2f} minutes.")
            else:
                print(f"       ❌ Failed with return code {result.returncode}. Check log for details.")
                
        except Exception as e:
            print(f"       ❌ Script execution error: {e}")
        
        print("-" * 50)

    print("\nAll experiments completed.")

if __name__ == "__main__":
    run_experiments()