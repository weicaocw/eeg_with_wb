import re
import os

def parse_chb_summary(summary_path):
    """
    解析 CHB-MIT 的 summary 文本文件。
    返回字典: { 'filename.edf': [(start_sec, end_sec), ...], ... }
    """
    seizure_info = {}
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    with open(summary_path, 'r') as f:
        content = f.read()

    # 按文件块分割
    # CHB summary 格式通常是: "File Name: chb01_01.edf" ... "Seizure Start Time: ..."
    # 我们使用简单的行遍历状态机来解析
    
    lines = content.split('\n')
    current_file = None
    
    for line in lines:
        line = line.strip()
        
        # 匹配文件名
        if line.startswith("File Name:"):
            current_file = line.split(":")[1].strip()
            if current_file not in seizure_info:
                seizure_info[current_file] = []
        
        # 匹配发作数量 (可选，用于校验)
        # 匹配具体时间
        # 格式可能是 "Seizure Start Time: 2996 seconds" 或 "Seizure 1 Start Time: ..."
        if "Seizure" in line and "Start Time" in line:
            # 提取数字
            try:
                start_t = int(re.search(r'(\d+)\s*seconds', line).group(1))
                # 寻找下一行或同一块的 End Time
                # 这里假设 Summary 文件的结构是固定的：Start 的下一行或几行后是 End
            except AttributeError:
                continue
                
            # 这是一个简化的逻辑，通常 End Time 在 Start Time 紧接着的几行
            # 为了更健壮，我们直接在当前上下文中找 End
            pass

    # 重新实现更稳健的正则匹配逻辑
    # 匹配整个块: File Name: (.*?) .*? Seizure Start Time: (\d+) .*? Seizure End Time: (\d+)
    # 注意：一个文件可能有多次发作
    
    file_pattern = re.compile(r"File Name:\s+(chb\d+_\d+\.edf)")
    start_pattern = re.compile(r"Seizure(?:\s+\d+)?\s+Start Time:\s+(\d+)\s+seconds")
    end_pattern = re.compile(r"Seizure(?:\s+\d+)?\s+End Time:\s+(\d+)\s+seconds")
    
    lines = content.split('\n')
    current_file = None
    
    for i, line in enumerate(lines):
        f_match = file_pattern.search(line)
        if f_match:
            current_file = f_match.group(1)
            if current_file not in seizure_info:
                seizure_info[current_file] = []
        
        if current_file:
            s_match = start_pattern.search(line)
            if s_match:
                start_time = int(s_match.group(1))
                # 尝试在接下来的几行找 End Time
                for j in range(1, 5): # 往下看5行
                    if i+j < len(lines):
                        e_match = end_pattern.search(lines[i+j])
                        if e_match:
                            end_time = int(e_match.group(1))
                            seizure_info[current_file].append((start_time, end_time))
                            break
                            
    return seizure_info