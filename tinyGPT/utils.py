import tomllib


def load_config(config_path: str = "config.toml") -> dict:
    """
    Load config data from TOML file

    Args:
        config_path: path of the config file to load

    Returns:
        dictionary of config data loaded from the file
    """
    with open(config_path, "rb") as f:
        data = tomllib.load(f)
    
    return data