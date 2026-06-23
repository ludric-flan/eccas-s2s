"""Multi-source acquisition: C3S, NMME (IRIDL), ERA5, CHIRPS, TAMSAT, gauges. [Phase 1]

All readers normalize to the canonical xarray layout (longitude/latitude, number,
decoded datetimes) defined in ``_base``.
"""
from eccas_s2s.io import nmme, c3s  # noqa: F401
from eccas_s2s.io.nmme import (  # noqa: F401
    build_nmme_url, download_nmme, open_nmme, load_nmme,
    NMME_MODELS, NMME_VARIABLES, decode_S,
)
from eccas_s2s.io.c3s import (  # noqa: F401
    build_c3s_request, download_c3s, download_c3s_forecast, download_c3s_hindcast,
    C3S_CENTRE_SYSTEMS, C3S_VARIABLES,
)
