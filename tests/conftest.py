from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


@pytest.fixture
def synthetic_crossmatch_path(tmp_path: Path) -> Path:
    flag_ltg = np.array([0, 1, 2, 3, 4, 4, 5], dtype=np.int16)
    flag_edgeon = np.array([0, 1, 2, 3, 4, 5, 5], dtype=np.int16)
    mp_ltg = np.array([0.1, 0.9, 0.2, 0.8, 0.01, 0.05, 0.95])
    mp_edgeon = np.array([0.0, 1.0, 0.2, 0.7, 0.03, 0.04, 0.4])
    offsets = np.array([-0.04, -0.02, 0.0, 0.02, 0.04])

    columns = [
        fits.Column(name="COADD_OBJECT_ID", format="8A", array=np.array([f"g{i}" for i in range(7)])),
        fits.Column(name="object_id", format="K", array=np.arange(100, 107)),
        fits.Column(name="RA_2", format="D", array=[10, 20, 30, 40, 50, 60, 70]),
        fits.Column(name="DEC_2", format="D", array=[-10, -5, 0, 5, 10, 15, 20]),
        fits.Column(name="MAG_AUTO_R", format="D", array=[19.8, 20, 20.2, 20.4, 20.6, 20.8, 21]),
        fits.Column(name="FLUX_RADIUS_R", format="D", array=[3, 4, 5, 6, 7, 8, 9]),
        fits.Column(name="MP_LTG", format="D", array=mp_ltg),
        fits.Column(name="MP_EdgeOn", format="D", array=mp_edgeon),
        fits.Column(name="FLAG_LTG", format="I", array=flag_ltg),
        fits.Column(name="FLAG_EdgeOn", format="I", array=flag_edgeon),
        fits.Column(name="Separation", format="D", unit="arcsec", array=[0, 0.1, 0.5, 0.9, 1, 0.2, 0.3]),
    ]
    for index, offset in enumerate(offsets, start=1):
        columns.append(fits.Column(name=f"P{index}_LTG", format="D", array=np.clip(mp_ltg + offset, 0, 1)))
        columns.append(fits.Column(name=f"P{index}_EdgeOn", format="D", array=np.clip(mp_edgeon + offset, 0, 1)))

    path = tmp_path / "synthetic_crossmatch.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns, name="Joined")]).writeto(path)
    return path
