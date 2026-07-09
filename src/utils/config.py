"""
===========================================================
PapsAI XNet Configuration Manager
===========================================================

This module provides utilities for loading, validating,
accessing, and saving configuration files used throughout
the PapsAI XNet framework.

Author:
Dr. William Wasswa

===========================================================
"""

from pathlib import Path
import yaml


class ConfigNode(dict):
    """
    Dictionary supporting attribute-style access.

    Example
    -------
    config.training.learning_rate
    """

    def __init__(self, dictionary):

        super().__init__()

        for key, value in dictionary.items():

            if isinstance(value, dict):
                value = ConfigNode(value)

            self[key] = value

    ########################################################

    def __getattr__(self, item):

        try:
            return self[item]

        except KeyError:

            raise AttributeError(item)

    ########################################################

    def __setattr__(self, key, value):

        self[key] = value

    ########################################################

    def to_dict(self):

        output = {}

        for key, value in self.items():

            if isinstance(value, ConfigNode):
                output[key] = value.to_dict()

            else:
                output[key] = value

        return output


############################################################


class ConfigManager:
    """
    Configuration manager for PapsAI XNet.
    """

    ########################################################

    def __init__(self, config_file):

        self.config_file = Path(config_file)

        if not self.config_file.exists():

            raise FileNotFoundError(

                f"Configuration file not found:\n{config_file}"
            )

        self.config = self._load()

    ########################################################

    def _load(self):

        with open(self.config_file, "r") as file:

            configuration = yaml.safe_load(file)

        return ConfigNode(configuration)

    ########################################################

    def reload(self):

        """
        Reload configuration.
        """

        self.config = self._load()

        return self.config

    ########################################################

    def save(self, output_file=None):

        """
        Save configuration.
        """

        if output_file is None:

            output_file = self.config_file

        output_file = Path(output_file)

        with open(output_file, "w") as file:

            yaml.dump(

                self.config.to_dict(),

                file,

                default_flow_style=False,

                sort_keys=False

            )

    ########################################################

    def update(self, key_path, value):

        """
        Update nested configuration.

        Example

        update(
            "training.learning_rate",
            0.0005
        )
        """

        keys = key_path.split(".")

        node = self.config

        for key in keys[:-1]:

            node = node[key]

        node[keys[-1]] = value

    ########################################################

    def print(self):

        """
        Pretty-print configuration.
        """

        print("=" * 60)

        print("PapsAI XNet Configuration")

        print("=" * 60)

        self._recursive_print(self.config)

    ########################################################

    def _recursive_print(self, dictionary, indent=0):

        for key, value in dictionary.items():

            if isinstance(value, ConfigNode):

                print(" " * indent + f"{key}")

                self._recursive_print(

                    value,

                    indent + 4

                )

            else:

                print(

                    " " * indent +

                    f"{key}: {value}"

                )


############################################################


def load_config(config_path="configs/config.yaml"):
    """
    Load configuration.

    Example

    config = load_config()

    lr = config.training.learning_rate
    """

    manager = ConfigManager(config_path)

    return manager.config


############################################################


def save_experiment_config(
        config,
        output_directory):

    """
    Save experiment configuration alongside
    trained models for reproducibility.
    """

    output_directory = Path(output_directory)

    output_directory.mkdir(

        parents=True,

        exist_ok=True

    )

    output_file = output_directory / "config.yaml"

    with open(output_file, "w") as file:

        yaml.dump(

            config.to_dict(),

            file,

            default_flow_style=False,

            sort_keys=False

        )
