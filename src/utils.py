from pathlib import Path


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


# Filepaths
ROOT = str(get_project_root())
DATA_PATH = ROOT + '/src/data/'
LOG_PATH = ROOT + '/src/logs/'

# Metrics
METRIC_LIST = ['balanced_accuracy', 'f1', 'roc_auc']
PARAM_LIST_PosScoringMatrix = ('p', 'q', 'C')
PARAM_LIST_SVC_linear = ('p', 'q', 'C')
PARAM_LIST_SVC_rbf = ('p', 'q', 'gamma')
PARAM_LIST_SVC_poly = ('p', 'q', 'gamma')
PARAM_LIST_SVC_sigmoid = ('p', 'q', 'gamma')
