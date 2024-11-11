import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file with environment overrides."""
        env = os.getenv("ENV", "dev")

        # Load base config
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # Load environment-specific config if it exists
        env_config_path = Path(self.config_path).parent / f"config.{env}.yaml"
        if env_config_path.exists():
            with open(env_config_path) as f:
                env_config = yaml.safe_load(f)
                config.update(env_config)

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
