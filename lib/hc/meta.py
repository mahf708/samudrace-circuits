"""Write meta.npz (lat, lon, per-output finite mask) for a capture dir from its fme ocean prediction file."""
import json, os, sys
import numpy as np, xarray as xr


def write(d):
    ds = xr.open_dataset(os.path.join(d, "fme", "ocean", "autoregressive_predictions.nc")).isel(sample=0, time=-1)
    names = json.load(open(os.path.join(d, "names.json")))["out_names"]
    mask = np.stack([np.isfinite(ds[n].values) for n in names])
    np.savez(os.path.join(d, "meta.npz"), lat=ds.lat.values, lon=ds.lon.values, out_mask=mask)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        write(d)
