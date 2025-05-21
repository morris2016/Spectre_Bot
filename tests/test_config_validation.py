import os
import pytest
from config import Config, load_config
from common.exceptions import ConfigurationError

def test_load_config_defaults(tmp_path):
    cfg_file = tmp_path / "config.json"
    cfg = load_config(str(cfg_file))
    assert isinstance(cfg, Config)
    cfg.validate()


def test_load_config_no_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config("config.json")
    assert isinstance(cfg, Config)
    assert os.path.isfile(tmp_path / "config.json")


def test_validate_missing_db_host():
    cfg = Config()
    cfg.data['database']['host'] = ''
    with pytest.raises(ConfigurationError):
        cfg.validate()
