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
T_lat_time = MarginalizeOutTransform(dims=["plev", "lon"]).fit_transform(T).T
T_plev_lat = MarginalizeOutTransform(dims=["time", "lon"]).fit_transform(T)
T_plev_time = MarginalizeOutTransform(dims=["lat", "lon"]).fit_transform(T).T

# plotfield = clif.visualization.plot_lat_lon(
#     cmap_name="e3sm_default",
#     print_stats=True,
#     title="T",
#     rhs_title=u"\u00b0" + "K",
#     lhs_title="e3sm ne30 pg2",
# )
# plotfield.show(T_lat_lon)

# plotfield2 = clif.visualization.plot_plev_lat(
#     cmap_name="e3sm_default",
#     title="T",
#     rhs_title=u"\u00b0" + "K",
# )
# plotfield2.show(T_plev_lat)

# plotfield3 = clif.visualization.plot_lat_time(
#     cmap_name="e3sm_default",
#     title="T",
#     rhs_title=u"\u00b0" + "K",
# )
# plotfield3.show(T_lat_time)

plotfield4 = clif.visualization.plot_plev_time(
    cmap_name="e3sm_default",
    title="T",
    rhs_title=u"\u00b0" + "K",
)
# plotfield4.show(T_plev_time)

weights = xr.open_dataarray("lat_lon_weights.nc")
pipe = Pipeline(
    steps=[
        ("anom", cpp.SeasonalAnomalyTransform()),
        (
            "marginalize",
            cpp.MarginalizeOutTransform(dims=["lat", "lon"], lat_lon_weights=weights),
        ),
        ("detrend", cpp.LinearDetrendTransform()),
        ("transpose", cpp.Transpose(dims=["plev", "time"])),
    ]
)

T_new = pipe.fit_transform(T)
plotfield5 = clif.visualization.plot_plev_time(
    cmap_name="e3sm_default_diff",
    title="T",
    rhs_title=u"\u00b0" + "K",
    log_plevs=True,
    nlevels=30,
)
# plotfield5.show(T_new)


# compute frequency/ period plots using fft
Ts = T_new.isel(plev=0)
time_index = Ts.indexes["time"]
time_diff_days = time_index[1:] - time_index[:-1]
avg_dt = np.mean([dti.days for dti in time_diff_days])
