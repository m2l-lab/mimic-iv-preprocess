import yaml

from common import load_config


def config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "paths": {"private_dir": "private", "report_dir": "report"},
        "data": {"mimic_root": "/configured/mimic"},
    }))
    return path


def test_config_can_be_read_without_creating_outputs(tmp_path):
    path = config_file(tmp_path)
    cfg = load_config(path, create_dirs=False)
    assert cfg["paths"]["private_dir"] == tmp_path / "private"
    assert cfg["paths"]["report_dir"] == tmp_path / "report"
    assert not (tmp_path / "private").exists()
    assert not (tmp_path / "report").exists()


def test_output_directories_are_created_with_private_permissions(tmp_path):
    cfg = load_config(config_file(tmp_path))
    assert cfg["paths"]["report_dir"].is_dir()
    assert cfg["paths"]["private_dir"].stat().st_mode & 0o777 == 0o700


def test_environment_overrides_do_not_change_explicit_config_precedence(tmp_path, monkeypatch):
    path = config_file(tmp_path)
    monkeypatch.setenv("TSL_CONFIG", str(path))
    monkeypatch.setenv("TSL_MIMIC_ROOT", "/override/mimic")
    cfg = load_config(create_dirs=False)
    assert cfg["data"]["mimic_root"] == "/override/mimic"
    assert cfg["paths"]["private_dir"] == path.parent / "private"
    monkeypatch.setenv("TSL_CONFIG", str(tmp_path / "missing.yaml"))
    assert load_config(path, create_dirs=False)["paths"] == cfg["paths"]
