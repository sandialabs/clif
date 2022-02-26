import xarray as xr
import dask
import os, sys
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline

try:
    import clif
except:
    sys.path.append("../")
    # from eof import fingerprints
    import clif

# import visualization tools and functions
import clif.visualization as cviz
import clif.preprocessing as cpp

DATA_DIR = "../../e3sm_data/fingerprint/"
T = xr.open_dataarray(os.path.join(DATA_DIR, "Temperature.nc"), chunks={"time": 1})

from clif.preprocessing import MarginalizeOutTransform, Transpose

T_lat_lon = MarginalizeOutTransform(dims=["plev", "time"]).fit_transform(T)

# plotfield = clif.visualization.plot_lat_lon(
#     cmap_name="e3sm_default",
#     print_stats=True,
#     title="T",
#     rhs_title=u"\u00b0" + "K",
#     lhs_title="e3sm ne30 pg2",
# )
# plotfield.show(T_lat_lon)

# T_lat_plev = MarginalizeOutTransform(dims=["time", "lon"]).fit_transform(T)
# plotfield2 = clif.visualization.plot_lat_plev(
#     cmap_name="e3sm_default",
#     title="T",
#     rhs_title=u"\u00b0" + "K",
# )
# plotfield2.show(T_lat_plev)

# T_lat_time = MarginalizeOutTransform(dims=["lon", "plev"]).fit_transform(T)
# plotfield3 = clif.visualization.plot_lat_time(
#     cmap_name="e3sm_default",
#     title="T",
#     rhs_title=u"\u00b0" + "K",
# )
# plotfield3.show(T_lat_time)

T = cpp.SeasonalAnomalyTransform().fit_transform(T)
T_time_plev = MarginalizeOutTransform(dims=["lat", "lon"]).fit_transform(T)
# T_plev_time = Transpose(dims=["plev", "time"]).fit_transform(T_time_plev)
# plotfield4 = clif.visualization.plot_time_plev(
#     cmap_name="e3sm_default",
#     title="T",
#     rhs_title=u"\u00b0" + "K",
# )
# plotfield4.draw(T_plev_time)
# plotfield4.add_colorbar()
# plotfield4.finish()

# pipe = Pipeline(
#     steps=[
#         ("anom", cpp.SeasonalAnomalyTransform()),
#         ("marginalize", cpp.MarginalizeOutTransform(dims=["lat", "lon"])),
#         # ("detrend", cpp.LinearDetrendTransform()),
#         ("transpose", cpp.Transpose(dims=["plev", "time"])),
#     ]
# )
# T_new = pipe.fit_transform(T)
plotfield5 = clif.visualization.plot_time_plev(
    cmap_name="e3sm_default_diff",
    title="T",
    rhs_title=u"\u00b0" + "K",
    log_plevs=False,
    nlevels=10,
)
plotfield5.draw(T_time_plev.T)
# plotfield5.finish()
