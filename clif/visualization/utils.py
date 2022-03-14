import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from .e3sm_cmap_colors import cet_rainbow, diverging_bwr, WBGYR

from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

####################################################
# Get colormaps
####################################################
def convert_to_cmap(rgb_array):
    rgb_arr = rgb_array / 255.0
    cmap = LinearSegmentedColormap.from_list(name="temp", colors=rgb_arr)
    return cmap


colormap_dict = {
    "e3sm_default": convert_to_cmap(cet_rainbow),
    "e3sm_default_diff": convert_to_cmap(diverging_bwr),
    "e3sm_precip_diff": plt.get_cmap("BrBG"),
    "e3sm_precip": convert_to_cmap(WBGYR),
    "e3sm_wind": plt.get_cmap("PiYG_r"),
}


def get_colormap(cmap_name):
    if cmap_name in colormap_dict.keys():
        return colormap_dict[cmap_name]
    else:
        try:
            return plt.get_cmap(cmap_name)
        except:
            print("CMAP name is not in any known lists.")


####################################################
# Misc Cartopy functions for plotting
####################################################
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
def add_cyclic(data, lat_name="lat", lon_name="lon"):
    lat, lon = data[lat_name], data[lon_name]
    data, lon = add_cyclic_point(data, coord=lon)
    return data, lat, lon
