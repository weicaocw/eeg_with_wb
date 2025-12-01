import tempfile
import os
import pytest
from src.utils.parser import parse_chb_summary

import os
import pytest
from src.utils.parser import parse_chb_summary

def test_parse_chb_summary_real_file():
    summary_path = "/tmp/wei_tmp/chb01/chb01-summary.txt"
    assert os.path.exists(summary_path), f"Summary file not found: {summary_path}"
    result = parse_chb_summary(summary_path)

    # 基本断言：至少有一个文件被解析
    assert isinstance(result, dict)
    assert len(result) > 0
    # 可选：检查某个具体文件和区间（根据实际数据可调整）
    example_file = next(iter(result.keys()))
    assert isinstance(result[example_file], list)
    # 检查区间格式
    if result[example_file]:
        assert isinstance(result[example_file][0], tuple)
        assert len(result[example_file][0]) == 2
        assert all(isinstance(x, int) for x in result[example_file][0])
        
    # Assert number of files 
    assert len(result) == 42 
    
    # Assert total number of seizure intervals
    total_intervals = sum(len(v) for v in result.values())
    assert total_intervals == 7
    
    # Assert chb01_03.edf has 1 seizure interval (2996, 3036)
    assert result.get("chb01_03.edf") == [(2996, 3036)]
    
    # Assert chb01_04.edf has 1 seizure interval (1467, 1494)
    assert result.get("chb01_04.edf") == [(1467, 1494)]
