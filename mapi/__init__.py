"""Market Anomaly Pressure Index (MAPI).

MAPI is an experimental research indicator. It is not a profitability claim and
must be validated out of sample before any practical use.
"""

from mapi.config import MapiConfig, load_config
from mapi.scoring import calculate_latest_mapi, calculate_mapi

__version__ = "0.2.0"

__all__ = [
    "MapiConfig",
    "calculate_latest_mapi",
    "calculate_mapi",
    "load_config",
    "__version__",
]
