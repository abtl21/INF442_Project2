from pathlib import Path


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


# Filepaths
ROOT = str(get_project_root())
DATA_PATH = ROOT + '/src/data/'
LOG_PATH = ROOT + '/src/logs/'
