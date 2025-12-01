#!/bin/bash

# ================= 配置区域 =================
# 1. 定义要运行的 Config 文件列表 (按要求的顺序)
CONFIG_FILES=(
    "config_super_xg.yaml"
    "config_noxwb_xg.yaml"
    "config_noxwt_xg.yaml"
    "config_noxwtxwb_xg.yaml"
    "config_noxwtxwbnomi_xg.yaml"

    "config_super_nonorm_xg.yaml"
    "config_noxwb_nonorm_xg.yaml"
    "config_noxwt_nonorm_xg.yaml"
    "config_noxwtxwb_nonorm_xg.yaml"
    "config_noxwtxwbnomi_nonorm_xg.yaml"

    "config_super_rf.yaml"
    "config_noxwb_rf.yaml"
    "config_noxwt_rf.yaml"
    "config_noxwtxwb_rf.yaml"
    "config_noxwtxwbnomi_rf.yaml"

    "config_super_nonorm_rf.yaml"
    "config_noxwb_nonorm_rf.yaml"
    "config_noxwt_nonorm_rf.yaml"
    "config_noxwtxwb_nonorm_rf.yaml"
    "config_noxwtxwbnomi_nonorm_rf.yaml"
)

# 2. 脚本和配置目录 (假设 main.py 在当前目录，且 configs 也在当前目录)
MAIN_SCRIPT="main.py"
CONFIG_DIR="./config/"

# ================= 无限循环运行逻辑 =================

# 使用 while true 开启一个无限循环
while true; do
    
    START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
    echo "========================================================"
    echo "🔄 Starting NEW CYCLE at $START_TIME"
    echo "========================================================"

    total=${#CONFIG_FILES[@]}

    # 循环遍历每个配置文件
    for i in "${!CONFIG_FILES[@]}"; do
        config_name="${CONFIG_FILES[$i]}"
        
        # 构造完整的配置路径
        FULL_CONFIG_PATH="${CONFIG_DIR}${config_name}"
        
        # 构造执行命令
        COMMAND="python $MAIN_SCRIPT --config $FULL_CONFIG_PATH"
        
        echo "[$(($i+1))/$total] ▶️ Running: $config_name"
        echo "CMD: $COMMAND"
        
        # 执行命令
        $COMMAND
        
        # 捕获上一个命令的退出状态
        EXIT_STATUS=$?
        
        if [ $EXIT_STATUS -eq 0 ]; then
            echo "       ✅ SUCCESS (Code 0): $config_name finished successfully."
        else
            # 打印错误信息
            echo "       ❌ ERROR (Code $EXIT_STATUS): $config_name FAILED!"
            echo "       Please check the terminal output above for the traceback."
            # 注意：这里仍然会继续运行下一个配置
        fi
        
        echo "--------------------------------------------------------"
    done

    END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
    echo "========================================================"
    echo "🟢 CYCLE COMPLETED. Starting next cycle immediately."
    echo "========================================================"
    
    # 循环将从 while true 顶部重新开始
done