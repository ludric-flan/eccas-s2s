"""
Shared helpers for the acquisition layer.

Every source module (c3s, nmme, era5, chirps, tamsat, obs_stations) downloads to a
local file and then normalizes the result to a canonical xarray layout so the rest of
the pipeline never has to care where the data came from.

Canonical conventions
----------------------
* spatial coords are named ``longitude`` / ``latitude`` (degrees east / north),
* ensemble member dimension is named ``number``,
* time coordinates are real (decoded) datetimes.
"""
import os
import urllib.request

# generic coordinate-name normalization shared by all sources
_COORD_RENAME = {
    "X": "longitude", "Y": "latitude",
    "lon": "longitude", "lat": "latitude",
    "M": "number",
}


def standardize_coords(ds):
    """Rename common coordinate variants to the canonical names, in place-safe way."""
    rename = {k: v for k, v in _COORD_RENAME.items() if k in ds.variables and v not in ds.variables}
    if rename:
        ds = ds.rename(rename)
    return ds


def http_download(url, dest, force_download=False, timeout=300):
    """
    Stream ``url`` to the local path ``dest``.

    Skips the download if ``dest`` already exists and is non-empty, unless
    ``force_download`` is set. Returns the path to the file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    if os.path.isfile(dest) and os.path.getsize(dest) > 0 and not force_download:
        print(f"  ✔ cached: {dest}")
        return dest

    print(f"  ⬇ downloading: {url}")
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "eccas-s2s/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    os.replace(tmp, dest)
    print(f"  ✔ saved: {dest} ({os.path.getsize(dest)} bytes)")
    return dest
