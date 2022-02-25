import xarray as xr
import dask
import os, sys
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

try:
    import clif
except:
    sys.path.append("../")
    # from eof import fingerprints
    import clif

# import visualization tools and functions
import clif.visualization as cviz

DATA_DIR = "../../e3sm_data/fingerprint/"
T = xr.open_dataarray(os.path.join(DATA_DIR, "Temperature.nc"), chunks={"time": 1})
T1_lat_lon = T.isel(time=0, plev=0)

cmap = cviz.get_colormap("e3sm_default")

plotfield = clif.visualization.plot_lat_lon_field(
    cmap_name="BrBG", print_stats=True, title="Temperature"
)
plotfield.draw_and_show(T1_lat_lon)
