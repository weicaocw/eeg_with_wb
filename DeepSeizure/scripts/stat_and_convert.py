import os
import csv
import json
from collections import Counter
import h5py
import mne
import warnings

# 配置路径
ROOT_DIR_EVAL = '/root/autodl-tmp/data/eval/'
OUTPUT_FILE_EVAL = 'eval_segments.json'
SAMPLING_RATE = 250.0
ROOT_DIR_TRAIN = '/root/autodl-tmp/data/train/'
OUTPUT_FILE_TRAIN = 'train_segments.json'
TARGET_ROOT_EVAL = '/root/autodl-tmp/data/h5/eval/'
TARGET_ROOT_TRAIN = '/root/autodl-tmp/data/h5/train/'

# 目标采样率
TARGET_SFREQ = 250.0

# 标准通道顺序 (严格有序)
STANDARD_CHANNELS = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'CZ'
]


def calculate_merged_seizure_duration(intervals):
    """
    计算所有 seizure 时间段的总时长（处理重叠）。
    intervals: list of tuple (start, stop)
    """
    if not intervals:
        return 0.0

    # 1. 按开始时间排序
    sorted_intervals = sorted(intervals, key=lambda x: x[0])

    merged = []
    for start, stop in sorted_intervals:
        if not merged:
            merged.append([start, stop])
        else:
            prev_start, prev_stop = merged[-1]
            if start < prev_stop:
                # 发生重叠或连接，更新结束时间为两者的最大值
                merged[-1][1] = max(prev_stop, stop)
            else:
                # 不重叠，添加新区间
                merged.append([start, stop])

    # 2. 计算合并后的总时长
    total_duration = sum(stop - start for start, stop in merged)
    return total_duration

def parse_csv_file(file_path):
    events = []
    seizure_intervals = [] # 用于计算总 seizure 时长
    total_duration_sec = 0.0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # 读取文件内容
        lines = f.readlines()
        
    # 解析 Header 获取 total_duration
    # Header 示例: # duration = 811.00 secs
    for line in lines:
        if line.startswith('#'):
            if 'duration' in line:
                try:
                    # 分割字符串找到数值部分
                    parts = line.split('=')
                    if len(parts) > 1:
                        dur_str = parts[1].strip().split()[0]
                        total_duration_sec = float(dur_str)
                except ValueError:
                    print(f"Warning: Could not parse duration in {file_path}")
    
    # 解析 CSV 数据部分
    # 找到非注释行的开始
    csv_lines = [line for line in lines if not line.startswith('#') and line.strip()]
    
    reader = csv.reader(csv_lines)
    
    # CSV 结构: channel, start_time, stop_time, label, confidence
    for row in reader:
        if len(row) < 4:
            continue
            
        channel_raw = row[0].strip()
        try:
            start_time = float(row[1].strip())
            stop_time = float(row[2].strip())
        except ValueError:
            continue # 跳过无法解析时间的行
            
        label = row[3].strip()
        
        # 解析 channel pair (例如 "T3-T5" -> ["T3", "T5"])
        # 有些特殊的 channel 名字可能不带 '-'，做一下兼容
        if '-' in channel_raw:
            pair = channel_raw.split('-')
        else:
            pair = [channel_raw] # 极端情况

        event_obj = {
            "pair": pair,
            "start_time": start_time,
            "stop_time": stop_time,
            "label": label
        }
        events.append(event_obj)
        
        # 收集非背景波的时间段用于计算时长
        if label != 'bckg':
            seizure_intervals.append((start_time, stop_time))
            
    return total_duration_sec, events, seizure_intervals

def stat_segments(root_dir, output_file, sampling_rate=250.0):
    metadata_list = []
    
    print(f"Scanning directory: {root_dir} ...")
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # 过滤文件：只处理 .csv 且忽略 _bi.csv
            if file.endswith('.csv') and not file.endswith('_bi.csv'):
                full_path = os.path.join(root, file)
                
                # 解析路径结构
                # 假设结构是: .../patient/session/montage/filename.csv
                # 例如: .../aaaaaqvx/s002_2015/01_tcp_ar/aaaaaqvx_s002_t001.csv
                
                # 获取相对路径 (相对于 eval 目录)
                rel_path = os.path.relpath(full_path, root_dir)
                path_parts = rel_path.split(os.sep)
                
                # 确保路径深度足够提取信息
                if len(path_parts) >= 3:
                    patient = path_parts[-3] # aaaaaqvx
                    session = path_parts[-2] # s002_2015
                    montage = path_parts[-1] # 01_tcp_ar (这里注意，通常 csv 在 montage 文件夹内)
                    # 如果 csv 直接在 session 下，可能需要调整索引，但根据你的示例，csv在montage文件夹内
                    # rel_path 示例: aaaaaqvx/s002_2015/01_tcp_ar/aaaaaqvx_s002_t001.csv
                    # path_parts: ['aaaaaqvx', 's002_2015', '01_tcp_ar', 'aaaaaqvx_s002_t001.csv']
                    
                    if len(path_parts) == 4:
                        patient = path_parts[0]
                        session = path_parts[1]
                        montage = path_parts[2]
                        filename = path_parts[3]
                    else:
                        # 简单的回退策略，或者根据实际情况调整
                        patient = path_parts[0]
                        session = path_parts[1] if len(path_parts) > 1 else "unknown"
                        montage = path_parts[2] if len(path_parts) > 2 else "unknown"
                        filename = file

                    segment_name = os.path.splitext(filename)[0]
                    file_path = f"{patient}/{session}/{montage}/{segment_name}" # 构建你要求的 file_path 格式

                    # 解析 CSV 内容
                    total_dur, events, sz_intervals = parse_csv_file(full_path)
                    
                    # 计算 Seizure 总时长 (合并重叠)
                    seizure_dur = calculate_merged_seizure_duration(sz_intervals)
                    
                    # 计算百分比
                    percentage = 0.0
                    if total_dur > 0:
                        percentage = (seizure_dur / total_dur) * 100
                    
                    # 计算数据点
                    data_points = int(total_dur * sampling_rate)
                    
                    # 构建元数据对象
                    meta = {
                        "patient": patient,
                        "session": session,
                        "segment": segment_name,
                        "montage": montage,
                        "total_duration_sec": total_dur,
                        "seizure_duration_sec": seizure_dur,
                        "percentage": percentage,
                        "data_points": data_points,
                        "events": events,
                        "file_path": file_path
                    }
                    
                    metadata_list.append(meta)

    # 输出 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=4)
        
    print(f"Successfully processed {len(metadata_list)} files.")
    print(f"Output saved to: {os.path.abspath(output_file)}")

class EEGSegment:
    """
    数据模型类：存储单个 EEG 片段的信息
    """
    def __init__(self, data):
        self.patient = data.get('patient', 'unknown')
        self.session = data.get('session', 'unknown')
        self.segment_name = data.get('segment', 'unknown')
        self.total_duration = data.get('total_duration_sec', 0.0)
        self.seizure_duration = data.get('seizure_duration_sec', 0.0)
        self.events = data.get('events', [])
        self.file_path = data.get('file_path', '') # e.g. "aaaa/s001/01_tcp/seg_name"
        
        # 唯一标识符：确保 session 统计时不会混淆（虽然 patient+session 通常唯一）
        self.session_id = f"{self.patient}_{self.session}"

    @property
    def has_seizure(self):
        """判断该片段是否包含 seizure 事件"""
        # 只要 seizure 持续时间大于 0 即视为有 seizure
        return self.seizure_duration > 0.0

    def __repr__(self):
        return f"<Segment: {self.segment_name}, Seizure: {self.has_seizure}>"

class EEGDataset:
    """
    数据集管理类：读取 JSON 并进行统计分析
    """
    def __init__(self, json_path):
        self.segments = []
        self.json_path = json_path
        self._load_data()

    def _load_data(self):
        """读取 JSON 并转换为对象列表"""
        print(f"正在读取文件: {self.json_path} ...")
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                raw_list = json.load(f)
                self.segments = [EEGSegment(item) for item in raw_list]
            print(f"成功加载 {len(self.segments)} 个片段数据。\n")
        except FileNotFoundError:
            print(f"错误: 找不到文件 {self.json_path}")
            self.segments = []
        except json.JSONDecodeError:
            print(f"错误: JSON 格式解析失败")
            self.segments = []

    def get_statistics(self):
        """执行统计逻辑"""
        if not self.segments:
            print("没有数据可统计。")
            return

        # 1. 基础集合 (用于去重统计)
        all_patients = set()
        all_sessions = set()
        
        # 2. Seizure 相关集合
        seizure_patients = set()
        seizure_sessions = set()
        seizure_segment_count = 0

        # 3. 聚合统计 (每个病人的片段数)
        patient_segment_counter = Counter()

        # 遍历列表进行统计
        for seg in self.segments:
            # 基础统计
            all_patients.add(seg.patient)
            all_sessions.add(seg.session_id)
            patient_segment_counter[seg.patient] += 1

            # Seizure 统计
            if seg.has_seizure:
                seizure_segment_count += 1
                seizure_patients.add(seg.patient)
                seizure_sessions.add(seg.session_id)

        # 4. 计算 Top 5 病人 (拥有最多 Segments)
        top_5_patients = patient_segment_counter.most_common(5)

        # --- 打印报告 ---
        print("=" * 40)
        print("EEG 数据集统计报告")
        print("=" * 40)
        
        print(f"{'统计维度':<20} | {'总数':<10} | {'含 Seizure 数量':<15} | {'Seizure 占比':<10}")
        print("-" * 65)
        
        # Segments
        total_segs = len(self.segments)
        seg_pct = (seizure_segment_count / total_segs * 100) if total_segs > 0 else 0
        print(f"{'Segments (片段)':<20} | {total_segs:<10} | {seizure_segment_count:<15} | {seg_pct:.2f}%")
        
        # Sessions
        total_sess = len(all_sessions)
        sess_count = len(seizure_sessions)
        sess_pct = (sess_count / total_sess * 100) if total_sess > 0 else 0
        print(f"{'Sessions (会话)':<20} | {total_sess:<10} | {sess_count:<15} | {sess_pct:.2f}%")
        
        # Patients
        total_pats = len(all_patients)
        pat_count = len(seizure_patients)
        pat_pct = (pat_count / total_pats * 100) if total_pats > 0 else 0
        print(f"{'Patients (病人)':<20} | {total_pats:<10} | {pat_count:<15} | {pat_pct:.2f}%")
        
        print("=" * 40)
        print("\n拥有最多 Segments 的前 5 位病人:")
        print("-" * 30)
        for rank, (patient, count) in enumerate(top_5_patients, 1):
            is_seizure_pat = "是" if patient in seizure_patients else "否"
            print(f"{rank}. 病人ID: {patient:<12} 片段数: {count:<5} (是否有癫痫: {is_seizure_pat})")
        print("-" * 30)

def normalize_channel_name(ch_name):
    """
    清洗通道名称，例如 'EEG FP1-REF' -> 'FP1'
    """
    clean = ch_name.upper().replace('EEG', '').strip()
    if '-' in clean:
        clean = clean.split('-')[0]
    return clean.strip()

def process_single_file(edf_path, output_root, source_root):
    """
    处理单个文件的流程（SSD 本地版）：Load -> Preprocess -> Save
    直接读取源文件并写入目标文件，不再经过 /tmp 中转。
    """
    try:
        # 1. 预先计算目标路径，确保目录存在
        # 计算相对路径: aaaaaqvx/s002_2015/01_tcp_ar/file.edf
        rel_path = os.path.relpath(edf_path, source_root)
        # 替换扩展名 .edf -> .h5
        rel_path_h5 = os.path.splitext(rel_path)[0] + '.h5'
        
        final_h5_path = os.path.join(output_root, rel_path_h5)
        
        # 确保目标子目录存在
        os.makedirs(os.path.dirname(final_h5_path), exist_ok=True)

        # 2. MNE 直接读取源文件 (preload=True 以便进行重采样和修改)
        # 忽略读取过程中的一些非致命警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 直接读取 edf_path
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='error')
        
        # 3. 通道匹配与重排
        original_names = raw.ch_names
        # 建立映射: 标准名 -> 原始名
        std_to_orig_map = {}
        
        # 遍历原始通道，尝试归一化并匹配
        for orig_ch in original_names:
            norm_name = normalize_channel_name(orig_ch)
            if norm_name in STANDARD_CHANNELS:
                std_to_orig_map[norm_name] = orig_ch
        
        # 检查完整性
        missing_channels = [ch for ch in STANDARD_CHANNELS if ch not in std_to_orig_map]
        if missing_channels:
            print(f"Skipping {os.path.basename(edf_path)}: Missing channels {missing_channels}")
            return False

        # 4. 提取并重排数据
        ordered_orig_names = [std_to_orig_map[std_ch] for std_ch in STANDARD_CHANNELS]
        
        # Pick 仅保留需要的通道
        raw.pick_channels(ordered_orig_names)
        
        # Reorder 强制按列表顺序排列
        raw.reorder_channels(ordered_orig_names)
        
        # 5. 重采样 (如果需要)
        if raw.info['sfreq'] != TARGET_SFREQ:
            # print(f"  Resampling {raw.info['sfreq']} -> {TARGET_SFREQ} Hz")
            raw.resample(TARGET_SFREQ, npad="auto")
            
        # 6. 获取 numpy 矩阵 [n_channels, n_times]
        data = raw.get_data()
        
        # 再次确认形状
        if data.shape[0] != 17:
            print(f"Error: Output channels mismatch. Expected 17, got {data.shape[0]}")
            return False

        # 7. 直接写入目标 H5 文件
        with h5py.File(final_h5_path, 'w') as f:
            # 兼容接口
            f.create_dataset('eeg', data=data, compression="gzip", compression_opts=4)
            
        return True

    except Exception as e:
        print(f"Failed to process {os.path.basename(edf_path)}: {e}")
        # 如果写入过程中出错，尝试清理可能生成的半成品文件
        if 'final_h5_path' in locals() and os.path.exists(final_h5_path):
            try: os.remove(final_h5_path)
            except: pass
        return False

def convert_edf_to_h5(source_root, target_root):
    print(f"Start converting EDF to H5...")
    print(f"Source: {source_root}")
    print(f"Target: {target_root}")
    
    success_count = 0
    fail_count = 0
    
    # 遍历源目录
    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file.endswith('.edf'):
                edf_full_path = os.path.join(root, file)
                
                # 执行转换
                if process_single_file(edf_full_path, target_root, source_root):
                    success_count += 1
                    if success_count % 10 == 0:
                        print(f"Processed {success_count} files...", end='\r')
                else:
                    fail_count += 1
    
    print(f"\nconversion finished!")
    print(f"Success: {success_count}")
    print(f"Failed/Skipped: {fail_count}")

if __name__ == "__main__":
    stat_segments(ROOT_DIR_EVAL, OUTPUT_FILE_EVAL, SAMPLING_RATE)
    stat_segments(ROOT_DIR_TRAIN, OUTPUT_FILE_TRAIN, SAMPLING_RATE)
    
    
    dataset_eval = EEGDataset(OUTPUT_FILE_EVAL) 
    dataset_train = EEGDataset(OUTPUT_FILE_TRAIN)
    
    convert_edf_to_h5(ROOT_DIR_EVAL, TARGET_ROOT_EVAL)
    convert_edf_to_h5(ROOT_DIR_TRAIN, TARGET_ROOT_TRAIN)
    
    