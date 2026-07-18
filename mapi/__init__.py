"""Market Anomaly Pressure Index (MAPI).

MAPI is an experimental research indicator. It is not a profitability claim and
must be validated out of sample before any practical use.
"""

from mapi.config import MapiConfig, load_config
from mapi.scoring import calculate_latest_mapi, calculate_mapi
from mapi.version import IMPLEMENTATION_VERSION

__version__ = IMPLEMENTATION_VERSION

__all__ = [
    "MapiConfig",
    "calculate_latest_mapi",
    "calculate_mapi",
    "load_config",
    "__version__",
]
