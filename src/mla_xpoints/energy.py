"""Energy axis for MLA spectra.

The files carry no calibration. ``Maximum Energy = 20 KV`` in the measurement settings and
2000 channels give a nominal 10 eV/channel from 0 eV. A least-squares fit of nine K lines
(Mg Ka to Fe Kb) in the summed spectrum of the RODARE 4896 test frame gives
``E = -10.04 eV + 10.0163 eV * channel`` with residuals within 3.3 eV. Pass
``offset_ev=NOMINAL_OFFSET_EV, gain_ev=NOMINAL_GAIN_EV`` for the nominal axis.
"""

from __future__ import annotations

import numpy as np

FITTED_OFFSET_EV = -10.04
FITTED_GAIN_EV = 10.0163
NOMINAL_OFFSET_EV = 0.0
NOMINAL_GAIN_EV = 10.0


def energy_axis_kev(
    n_channels: int = 2000, offset_ev: float = FITTED_OFFSET_EV, gain_ev: float = FITTED_GAIN_EV
) -> np.ndarray:
    """Energy (keV) at each channel index ``0..n_channels-1``."""
    if n_channels <= 0:
        raise ValueError("n_channels must be positive")
    return (offset_ev + gain_ev * np.arange(n_channels)) / 1000.0
