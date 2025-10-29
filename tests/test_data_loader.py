import pandas as pd
from my_pipeline.data_loader import readers

def test_load_data_simple(tmp_path):
    # 准备：创建一个临时的测试文件
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.csv"
    p.write_text("col1,col2\n1,a\n2,b")

    # 执行：调用被测试的函数
    data = readers.load_data(str(p))

    # 断言：检查结果是否符合预期
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 2
    assert data['col1'].iloc[0] == 1
