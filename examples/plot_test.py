import xarray as xr
import dask
import os
from dataclasses import dataclass, field

from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import numpy.ma as ma
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

import matplotlib
import matplotlib.colors as colors  # isort:skip  # noqa: E402
import matplotlib.pyplot as plt  # isort:skip  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap

DATA_DIR = "../../e3sm_data/fingerprint/"
T = xr.open_dataarray(os.path.join(DATA_DIR, "Temperature.nc"), chunks={"time": 1})
T1_lat_lon = T.isel(time=0, plev=0)

#### MISC Functions
def get_ax_size(fig, ax):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height


def determine_tick_step(degrees_covered):
    if degrees_covered > 180:
        return 60
    if degrees_covered > 60:
        return 30
    elif degrees_covered > 30:
        return 10
    elif degrees_covered > 20:
        return 5
    else:
        return 1


# add cyclic point to longitude
def add_cyclic(data):
    lat, lon = data["lat"], data["lon"]
    data, lon = add_cyclic_point(data, coord=lon)
    return data, lat, lon


colormap_dict = {
    "e3sm_precip": "WhiteBlueGreenYellowRed.rgb",
    "e3sm_default": "cet_rainbow.rgb",
    "e3sm_default_diff": "diverging_bwr.rgb",
    "e3sm_precip_diff": plt.get_cmap("BrBG"),
    "e3sm_wind": plt.get_cmap("PiYG_r"),
}

# get color map
def get_colormap(cmapstyle=colormap_dict["e3sm_default"]):
    colormap = cmapstyle
    if type(colormap) == str:
        rgb_arr = np.loadtxt(colormap)
        rgb_arr = rgb_arr / 255.0
        cmap = LinearSegmentedColormap.from_list(name=colormap, colors=rgb_arr)
    else:
        cmap = cmapstyle
    return cmap


plotTitle = {"fontsize": 11.5}
plotSideTitle = {"fontsize": 9.5}

# panel = [
#     (0.1691, 0.6810, 0.6465, 0.2258),
#     (0.1691, 0.3961, 0.6465, 0.2258),
#     (0.1691, 0.1112, 0.6465, 0.2258),
# ]
panel = [
    (0.1, 0.1, 0.75, 0.8),
    # (0.1691, 0.3961, 0.6465, 0.2258),
    # (0.1691, 0.1112, 0.6465, 0.2258),
]
#######################

data, lat, lon = add_cyclic(T1_lat_lon)

# Setting up for plotting
cmap = get_colormap()
proj = ccrs.PlateCarree()

data_mean = data.mean()
data_min = data.min()
data_max = data.max()
stats = [data_mean, data_min, data_max]

clevels = []  # contour levels


@dataclass
class lat_lon_params:
    regions: list = field(default_factory=lambda: ["global"])
    figsize: list = field(default_factory=lambda: [8.5, 4.0])
    dpi: int = 150


parameters = lat_lon_params()

regions_specs = {"global": {}}


def plot_panel(
    n,
    fig,
    proj,
    var,
    lat,
    lon,
    clevels,
    cmap,
    title,
    parameters,
    stats,
):

    # var = add_cyclic(var)
    lon = np.array(lon)  # var.getLongitude()
    lat = lat.values  # var.getLatitude()
    var = ma.squeeze(var)

    # Contour levels
    levels = None
    norm = None
    # if len(clevels) > 0:
    levels = var.min() + (var.max() - var.min()) * np.linspace(
        0, 1, 30
    )  # [-1.0e8] + clevels + [1.0e8]
    #     norm = colors.BoundaryNorm(boundaries=levels, ncolors=256)

    # ax.set_global()
    region_str = parameters.regions[0]
    region = regions_specs[region_str]
    global_domain = True
    full_lon = True
    # if "domain" in region.keys():  # type: ignore
    #     # Get domain to plot
    #     domain = region["domain"]  # type: ignore
    #     global_domain = False
    # else:
    #     # Assume global domain
    #     domain = cdutil.region.domain(latitude=(-90.0, 90, "ccb"))
    # kargs = domain.components()[0].kargs
    lon_west, lon_east, lat_south, lat_north = (
        0,
        360,
        -90,
        90,
    )
    # if "longitude" in kargs:
    #     full_lon = False
    #     lon_west, lon_east, _ = kargs["longitude"]
    #     # Note cartopy Problem with gridlines across the dateline:https://github.com/SciTools/cartopy/issues/821. Region cross dateline is not supported yet.
    #     if lon_west > 180 and lon_east > 180:
    #         lon_west = lon_west - 360
    #         lon_east = lon_east - 360

    # if "latitude" in kargs:
    #     lat_south, lat_north, _ = kargs["latitude"]
    lon_covered = lon_east - lon_west
    lon_step = determine_tick_step(lon_covered)
    xticks = np.arange(lon_west, lon_east, lon_step)
    # Subtract 0.50 to get 0 W to show up on the right side of the plot.
    # If less than 0.50 is subtracted, then 0 W will overlap 0 E on the left side of the plot.
    # If a number is added, then the value won't show up at all.
    if global_domain or full_lon:
        xticks = np.append(xticks, lon_east - 0.50)
        proj = ccrs.PlateCarree(central_longitude=180)
    else:
        xticks = np.append(xticks, lon_east)
    lat_covered = lat_north - lat_south
    lat_step = determine_tick_step(lat_covered)
    yticks = np.arange(lat_south, lat_north, lat_step)
    yticks = np.append(yticks, lat_north)
    yticks[0], yticks[-1] = np.ceil(lat.min()), np.ceil(lat.max())

    # Contour plot
    ax = fig.add_axes(panel[n], projection=proj)
    # ax = plt.axes(projection=proj)
    # ax.set_extent([lon_west, lon_east, lat_south, lat_north], crs=proj)
    # cmap = get_colormap(cmap, parameters)
    p1 = ax.contourf(
        lon,
        lat,
        var,
        transform=ccrs.PlateCarree(),
        norm=None,
        levels=30,
        cmap=cmap,
        extend="both",
    )

    # ax.set_aspect("auto")
    # Full world would be aspect 360/(2*180) = 1
    ax.set_aspect((lon_east - lon_west) / (2 * (lat_north - lat_south)))
    ax.coastlines(lw=0.5)
    if not global_domain and "RRM" in region_str:
        ax.coastlines(resolution="50m", color="black", linewidth=1)
        state_borders = cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_1_states_provinces_lakes",
            scale="50m",
            facecolor="none",
        )
        ax.add_feature(state_borders, edgecolor="black")
    if title[0] is not None:
        ax.set_title(title[0], loc="left", fontdict=plotSideTitle)
    if title[1] is not None:
        ax.set_title(title[1], fontdict=plotTitle)
    if title[2] is not None:
        ax.set_title(title[2], loc="right", fontdict=plotSideTitle)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True, number_format=".0f")
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=8.0, direction="out", width=1)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    # Color bar
    cbax = fig.add_axes((0.89, 0.12, 0.0326, 0.66))
    cbar = fig.colorbar(p1, cax=cbax)
    w, h = get_ax_size(fig, cbax)

    if levels is None:
        cbar.ax.tick_params(labelsize=9.0, length=0)

    else:
        cmap_thin = 2
        maxval = np.amax(np.absolute(levels[1:-1]))
        if maxval < 10.0:
            fmt = "%5.2f"
            pad = 26
        elif maxval < 100.0:
            fmt = "%5.2f"
            pad = 29
        else:
            fmt = "%6.2f"
            pad = 34
        cbar.set_ticks(levels[1:-1:cmap_thin])
        labels = [fmt % level for level in levels[1:-1:cmap_thin]]
        cbar.ax.set_yticklabels(labels, ha="right")
        cbar.ax.tick_params(labelsize=9.0, pad=pad, length=0)

    # Min, Mean, Max
    fig.text(
        0.87,
        0.22 + 0.6,
        "Max\nMean\nMin",
        ha="left",
        fontdict=plotSideTitle,
    )
    fig.text(
        0.98,
        0.22 + 0.6,
        "{0:.2f}\n{1:.2f}\n{2:.2f}".format(*stats[0:3]),
        ha="right",
        fontdict=plotSideTitle,
    )

    # # RMSE, CORR
    # if len(stats) == 5:
    #     fig.text(
    #         panel[n][0] + 0.6635,
    #         panel[n][1] - 0.0105,
    #         "RMSE\nCORR",
    #         ha="left",
    #         fontdict=plotSideTitle,
    #     )
    #     fig.text(
    #         panel[n][0] + 0.7635,
    #         panel[n][1] - 0.0105,
    #         "{0:.2f}\n{1:.2f}\n{2:.2f}".format(*stats[0:3]),
    #         ha="right",
    #         fontdict=plotSideTitle,
    #     )

    # # grid resolution info:
    # if n == 2 and "RRM" in region_str:
    #     dlat = lat[2] - lat[1]
    #     dlon = lon[2] - lon[1]
    #     fig.text(
    #         panel[n][0] + 0.4635,
    #         panel[n][1] - 0.04,
    #         "Resolution: {:.2f}x{:.2f}".format(dlat, dlon),
    #         ha="left",
    #         fontdict=plotSideTitle,
    #     )


n, proj = 0, proj
var, clevels, cmap = data, clevels, cmap
title = [None, "Temperature", None]

fig = plt.figure(figsize=parameters.figsize, dpi=parameters.dpi)

plot_panel(
    n,
    fig,
    proj,
    var,
    lat,
    lon,
    clevels,
    cmap,
    title,
    parameters,
    stats=stats,
)

# fig.savefig("test.png", bbox_inches="tight")
plt.show()
