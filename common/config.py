from pathlib import Path

import yaml


def load_config() -> dict[str, str]:
  config_path = Path(__file__).parent.parent / "config.yaml"
  with config_path.open(encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

  if not isinstance(config, dict) or not all(
    isinstance(config.get(key), str) and config[key]
    for key in ("password", "session_secret")
  ):
    raise RuntimeError("config.yaml must define non-empty password and session_secret values")

  return config