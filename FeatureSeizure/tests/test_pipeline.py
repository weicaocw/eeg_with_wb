import configparser
from my_pipeline import pipeline

def test_pipeline_run(tmp_path):
    # 准备：创建测试用的数据和配置文件
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_file = data_dir / "test.csv"
    data_file.write_text("col1,col2\n1,a\n2,b")

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # 创建一个测试配置文件
    test_conf = configparser.ConfigParser()
    test_conf['paths'] = {
        'data_input_file': str(data_file),
        'results_output_dir': str(results_dir)
    }
    test_conf['pipeline_modules'] = {
        'run_module_1': 'True',
        'run_module_2': 'False'
    }
    conf_path = tmp_path / "test.conf"
    with open(conf_path, 'w') as f:
        test_conf.write(f)

    # 加载配置对象
    cfg = configparser.ConfigParser()
    cfg.read(conf_path)

    # 执行：运行整个（或部分）流程
    pipeline.run(cfg)

    # 断言：检查集成结果（例如，输出文件是否已创建）
    # (假设 module_1 会创建一个 'mod1_results.json')
    output_file = results_dir / "mod1_results.json"
    assert output_file.exists()
