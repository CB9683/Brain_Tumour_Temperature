# tests/test_config_manager.py
import pytest
import yaml
import os
from src import config_manager
import logging

# Fixture to create a temporary valid config file
@pytest.fixture
def temp_valid_config_file(tmp_path):
    config_data = {
        "paths": {"output_dir": "test_output"},
        "simulation": {"random_seed": 123, "log_level": "DEBUG"},
        "nested_params": {"level1": {"level2": "value"}}
    }
    config_file = tmp_path / "valid_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    return str(config_file), config_data

# Fixture to create a temporary malformed config file
@pytest.fixture
def temp_malformed_config_file(tmp_path):
    config_file = tmp_path / "malformed_config.yaml"
    with open(config_file, 'w') as f:
        f.write("paths: {output_dir: test_output\nlog_level: INFO") # Malformed YAML
    return str(config_file)

def test_load_config_valid(temp_valid_config_file):
    config_path, expected_data = temp_valid_config_file
    config = config_manager.load_config(config_path)
    assert config == expected_data
    assert config_manager.get_param(config, "simulation.random_seed") == 123

def test_load_config_non_existent():
    with pytest.raises(FileNotFoundError):
        config_manager.load_config("non_existent_config.yaml")

def test_load_config_malformed(temp_malformed_config_file):
    config_path = temp_malformed_config_file
    with pytest.raises(yaml.YAMLError):
        config_manager.load_config(config_path)

def test_get_param_existing(temp_valid_config_file):
    config_path, config_data = temp_valid_config_file
    config = config_manager.load_config(config_path)
    
    assert config_manager.get_param(config, "paths.output_dir") == "test_output"
    assert config_manager.get_param(config, "simulation.random_seed") == 123
    assert config_manager.get_param(config, "nested_params.level1.level2") == "value"

def test_get_param_non_existent(temp_valid_config_file):
    config_path, _ = temp_valid_config_file
    config = config_manager.load_config(config_path)

    assert config_manager.get_param(config, "non.existent.key") is None
    assert config_manager.get_param(config, "non.existent.key", "default_val") == "default_val"
    assert config_manager.get_param(config, "simulation.non_existent", 999) == 999

def test_create_default_config_new_file(tmp_path):
    default_config_path = tmp_path / "default_config.yaml"
    assert not os.path.exists(default_config_path)
    
    config_manager.create_default_config(str(default_config_path))
    assert os.path.exists(default_config_path)
    
    # Try loading it to ensure it's valid YAML
    try:
        loaded_default_config = config_manager.load_config(str(default_config_path))
        assert "paths" in loaded_default_config # Check a known top-level key
        assert "output_dir" in loaded_default_config["paths"]
    except Exception as e:
        pytest.fail(f"Loading default config failed: {e}")


def test_create_default_config_existing_file(tmp_path, caplog):
    default_config_path = tmp_path / "existing_default_config.yaml"
    # Create a dummy file first
    with open(default_config_path, "w") as f:
        f.write("some: content")
    
    # Temporarily set the logging level for the relevant logger for this test
    # This ensures INFO messages are captured.
    # You can also set this globally in pytest.ini or pyproject.toml if needed for many tests.
    with caplog.at_level(logging.INFO, logger="src.config_manager"): # Specify the logger name
        config_manager.create_default_config(str(default_config_path))
    
    # Check that the file was not overwritten and a log message was produced
    with open(default_config_path, "r") as f:
        content = f.read()
        assert content == "some: content"
    
    print(f"Captured log text: {caplog.text}") # For debugging if it still fails
    assert f"Configuration file already exists: {str(default_config_path)}" in caplog.text