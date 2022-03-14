from email.mime import base
from shutil import which
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
import xarray

from .utils import add_cyclic, get_colormap, determine_tick_step, get_ax_size

from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

# (left, bottom, width, height) for plotting map
panel = (0.1, 0.1, 0.75, 0.8)

# range of lat and lon coords
lon_west, lon_east, lat_south, lat_north = (
    0,
    360,
    -90,
    90,
)

### Jake's plotting script
def plot_field(
    EOFs,
    eof_to_print: int,
    lats: list,
    lons: list,
    cmap: matplotlib.colors.Colormap = plt.get_cmap("cividis"),
    ax: plt.axes = None,
    title: str = "",
    grid: bool = False,
    colorbar_title: bool = "",
    grid_kwargs: dict = {},
) -> tuple:
    """Plots a given fingerprint's EOFs as a 2-dimensional field on a latitude by longitude grid.

    Parameters
    ----------
    eof_to_print : int
        The specific EOF to print in order of variance explained in the un-rotated set.
    lats : list
        List of latitude values.
    lons : list
        List of longitude values.
    cmap : matplotlib.pyplot.cmap, optional
        A given colormap, by default plt.get_cmap("jet")
    ax : matplotlib.pyplot.axes, optional
        Given matplotlib axes, by default None
    title : str, optional
        Title of plot, by default ""
    grid : bool, optional
        Select whether a lattitude by longtiude grid is plotted, by default False
    colorbar_title : str, optional
        Title of colorbar, by default ""
    grid_kwargs : dict, optional
        Alternative arguments for the grid layout, by default {}

    Returns
    -------
    tuple
        A tuple of the plot's figure, axes, and colorbar objects.
    """
    EOF_recons = np.reshape(EOFs_[eof_to_print], (len(lats), len(lons)))
    data = EOF_recons  # [eof_to_print, :, :]

    if not ax:
        f = plt.figure(figsize=(8, (data.shape[0] / float(data.shape[1])) * 8))
        ax = plt.axes(projection=ccrs.PlateCarree())

    data, lons = add_cyclic_point(data, coord=lons)
    pl = plt.contourf(
        lons, lats, data, cmap=cmap, extend="both", transform=ccrs.PlateCarree()
    )
    ax.coastlines()
    _colorbar = plt.colorbar(pl, label=colorbar_title)
    if grid:
        if grid_kwargs:
            ax.gridlines(**grid_kwargs)
        else:
            ax.gridlines(
                draw_labels=True,
                dms=True,
                x_inline=False,
                y_inline=False,
                linestyle="--",
                color="black",
            )
    ax.set_title(title, fontsize=16)

    return f, ax, _colorbar


from abc import ABC, abstractmethod


class BasePlot(ABC):
    @abstractmethod
    def draw(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_yaxis_properties(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_xaxis_properties(self, *args, **kwargs):
        pass

    @abstractmethod
    def finish(self, *args, **kwargs):
        pass

    @abstractmethod
    def show(self, *args, **kwargs):
        pass


class BaseContourPlot(BasePlot):
    def __init__(
        self,
        cmap_name="e3sm_default",
        proj=None,
        figsize=[8.5, 4.0],
        region=None,
        dpi=150,
        print_stats=True,
        nlevels=30,
        panel=panel,
        title=None,
        rhs_title=None,
        lhs_title=None,
        show_full=True,
        title_fontdict={"fontsize": 14.0},
        sidetitle_fontdict={"fontsize": 7.5},
    ):
        self.cmap_name = cmap_name
        self.proj = proj
        self.figsize = figsize
        self.region = region
        self.dpi = dpi
        self.print_stats = print_stats
        self.nlevels = nlevels
        self.panel = panel
        self.title = title
        self.rhs_title = rhs_title
        self.lhs_title = lhs_title
        self.show_full = show_full
        self.plotTitle = title_fontdict
        self.plotSideTitle = sidetitle_fontdict

    def get_coords_from_data_array(self, data, dim):

        if dim == "time":
            # return a datetime array instead of cftime
            x = data.indexes[dim].to_datetimeindex(unsafe=True)
            data[dim] = x
        elif dim == "lon" or dim == "longitude":
            data_ma, x = add_cyclic_point(data, coord=data[dim])
            x = np.array(x)
            lat_dim = list(set(data.dims) - set([dim]))[0]
            data_temp = xarray.DataArray(np.array(data_ma), dims=[lat_dim, dim])
            data = data_temp.assign_coords({lat_dim: data[lat_dim], dim: x})
        else:
            # lat, plev, etc., no need for custom data extraction
            x = np.array(data[dim])
        return x, data

    def compute_stats(self, data):
        if isinstance(data, xarray.DataArray):
            data = data.values
        stats = {"Min": data.min(), "Mean": data.mean(), "Max": data.max()}
        return stats

    def compute_contour_levels(self, data):
        if isinstance(data, xarray.DataArray):
            data_np = data.values
        levels_ = data.min() + (data.max() - data.min()) * np.linspace(
            0, 1, self.nlevels
        )
        return levels_

    def show_titles(self):
        assert hasattr(self, "ax_"), "Must draw object first."
        if self.lhs_title is not None:
            self.ax_.set_title(self.lhs_title, loc="left", fontdict=self.plotSideTitle)
        if self.title is not None:
            self.ax_.set_title(self.title, fontdict=self.plotTitle)
        if self.rhs_title is not None:
            self.ax_.set_title(self.rhs_title, loc="right", fontdict=self.plotSideTitle)

    def draw(
        self,
        data: xarray.DataArray,
        x_name: str,
        y_name: str,
        fig=None,
        ax=None,
        **draw_params,
    ):

        # retrieve coordinate data
        self.x_, data_ = self.get_coords_from_data_array(data, dim=x_name)
        self.y_, data_ = self.get_coords_from_data_array(data_, dim=y_name)
        y, x, var = self.y_, self.x_, data_.values

        self.stats_dict_ = self.compute_stats(data_)

        # Contour levels
        self.levels_ = self.compute_contour_levels(var)

        if fig is None and ax is None:
            """Create figure if none is given"""
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        if ax is None:
            # Add a map projection
            ax = fig.add_axes(self.panel, projection=self.proj)

        # get cmap
        self.cmap_ = get_colormap(self.cmap_name)

        # plot contour map
        cplt = ax.contourf(
            x,
            y,
            var,
            transform=self.proj,
            norm=None,
            levels=self.levels_,
            cmap=self.cmap_,
            extend="both",
        )

        self.ax_, self.fig_ = ax, fig
        self.cplt_ = cplt

    def add_colorbar(self, rect=(0.89, 0.14, 0.0326, panel[3] - 0.08)):
        """draw color bar relative to location of map graph

        rect = (left, bottom, width, height)
        see self.panel for reference

        """
        if not self.print_stats:
            rect = (
                0.89,
                0.12,
                0.0326,
            )

        assert hasattr(self, "cplt_"), "Must run draw() to generate a contour plot"

        # Color bar
        fig = self.fig_
        cbax = fig.add_axes(rect)
        cbar = fig.colorbar(self.cplt_, cax=cbax, drawedges=True, alpha=0.5)
        cbar.outline.set_alpha(0.75)
        cbar.dividers.set_alpha(0.6)
        # cbar.solids.set_edgecolor("k", alpha=0.2)
        w, h = get_ax_size(fig, cbax)

        if self.levels_ is None:
            cbar.ax.tick_params(labelsize=8.0, length=0)

        else:
            cmap_thin = 2
            maxval = np.amax(np.absolute(self.levels_[1:-1]))
            if maxval < 10.0:
                fmt = "%5.2f"
                pad = 26
            elif maxval < 100.0:
                fmt = "%5.2f"
                pad = 29
            else:
                fmt = "%6.2f"
                pad = 34
            cbar.set_ticks(self.levels_[1:-1:cmap_thin])
            labels = [fmt % level for level in self.levels_[1:-1:cmap_thin]]
            cbar.ax.set_yticklabels(labels, ha="right")
            cbar.ax.tick_params(labelsize=8.0, pad=pad, length=0)

    def finish(self, custom=False, **finish_params):
        ax, fig = self.ax_, self.fig_
        x, y = self.x_, self.y_

        # print titles
        self.show_titles()

        self.show_stats()

        if not custom:
            self.set_xaxis_properties()
            self.set_yaxis_properties()

    def set_yaxis_properties(self):
        pass

    def set_xaxis_properties(self):
        pass

    def set_axis_to_longitude(self, which_axis="x", proj=ccrs.PlateCarree()):
        ax = self.ax_
        # set lon axis
        lon_covered = lon_east - lon_west
        lon_step = determine_tick_step(lon_covered)
        ticks = np.arange(lon_west, lon_east, lon_step)
        ticks = np.append(ticks, lon_east - 0.50)

        ax.set_xticks(ticks, crs=proj)
        lon_formatter = LongitudeFormatter(
            zero_direction_label=True, number_format=".0f"
        )
        if which_axis == "x":
            ax.xaxis.set_major_formatter(lon_formatter)
            ax.xaxis.set_ticks_position("bottom")
        elif which_axis == "y":
            ax.yaxis.set_major_formatter(lon_formatter)
            ax.yaxis.set_ticks_position("left")

    def set_axis_to_latitude(self, which_axis="y", proj=ccrs.PlateCarree()):
        ax = self.ax_
        if which_axis == "x":
            lat = self.x_
        elif which_axis == "y":
            lat = self.y_
        lat_covered = lat_north - lat_south
        lat_step = determine_tick_step(lat_covered)
        ticks = np.arange(lat_south, lat_north, lat_step)
        ticks = np.append(ticks, lat_north)
        if not self.show_full:
            ticks[0], ticks[-1] = np.ceil(lat.min()), np.ceil(lat.max())
        lat_formatter = LatitudeFormatter()

        if which_axis == "x":
            ax.set_xticks(ticks, crs=proj)
            ax.xaxis.set_major_formatter(lat_formatter)
            ax.xaxis.set_ticks_position("bottom")
        elif which_axis == "y":
            ax.set_yticks(ticks, crs=proj)
            ax.yaxis.set_major_formatter(lat_formatter)
            ax.yaxis.set_ticks_position("left")
        ax.tick_params(labelsize=8.0, direction="out", width=1)

        return None

    def set_axis_to_plev(self, which_axis="y"):
        axis = which_axis
        ax = self.ax_

        getattr(ax, f"set_{axis}label")("pressure (Pa)")
        if self.log_plevs:
            getattr(ax, f"set_{axis}scale")("log")
        getattr(ax, f"invert_{axis}axis")()

    def set_axis_to_time(self, which_axis="x", label=None):
        if which_axis == "x":
            ax = self.ax_
            if label is not None:
                ax.set_xlabel(label)
            ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
            ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator((1, 4, 7, 10)))
        else:
            raise NotImplementedError

    def show_stats(self):
        # add stats on top of colorbar
        # colorbar panel for reference:
        #   colorbar_rect=(0.89, 0.12, 0.0326, 0.66)
        stat_names = self.stats_dict_.keys()
        stat_values = np.array(list(self.stats_dict_.values()))
        stat_values = np.array(list(self.stats_dict_.values()))
        stat_values = ["{0:.2f}".format(v) for v in stat_values]
        if self.print_stats:
            # Min, Mean, Max
            self.fig_.text(
                0.88,
                panel[3] + 0.08,
                "\n".join(stat_names),
                ha="left",
                fontdict=self.plotSideTitle,
            )
            self.fig_.text(
                0.97,
                panel[3] + 0.08,
                "\n".join(stat_values),
                ha="right",
                fontdict=self.plotSideTitle,
            )

    def show(
        self,
        data: xarray.DataArray,
        fig=None,
        save=False,
        file_name=None,
        draw_params={},
        colorbar_params={},
        finish_params={},
    ):
        self.draw(data, fig=fig, **draw_params)
        self.add_colorbar(**colorbar_params)
        self.finish(**finish_params)
        if save:
            assert file_name is not None, "Must provide a valid file name"
            self.fig_.savefig(file_name)
        plt.show()
        return self


class plot_lat_lon(BaseContourPlot):
    def __init__(
        self,
        proj=ccrs.PlateCarree(central_longitude=180.0),
        region="global",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.proj = proj
        self.region = region

    def draw(
        self, data: xarray.DataArray, x_name="lon", y_name="lat", fig=None, ax=None
    ):
        super().draw(data, x_name=x_name, y_name=y_name)
        self.ax_.coastlines(lw=0.35)

    def set_yaxis_properties(self):
        self.set_axis_to_latitude()

    def set_xaxis_properties(self):
        self.set_axis_to_longitude(proj=ccrs.PlateCarree(central_longitude=0.0))

    def finish(self, custom=True):

        super().finish(custom=custom)

        gridliner = self.ax_.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=True, alpha=0.2
        )
        gridliner.top_labels = False
        gridliner.right_labels = False

        return None


class plot_plev_lat(BaseContourPlot):
    def __init__(self, proj=None, show_full=False, log_plevs=True, *args, **kwargs):
        super().__init__(proj=proj, show_full=show_full, *args, **kwargs)
        self.log_plevs = log_plevs

    def draw(
        self, data: xarray.DataArray, x_name="lat", y_name="plev", fig=None, ax=None
    ):
        super().draw(data, x_name=x_name, y_name=y_name)
        self.ax_.set_aspect("auto")

    def set_yaxis_properties(self):
        self.set_axis_to_plev()

    def set_xaxis_properties(self):
        self.set_axis_to_latitude(which_axis="x")


class plot_lat_time(BaseContourPlot):
    def __init__(self, proj=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj = proj

    def set_yaxis_properties(self):
        self.set_axis_to_latitude()

    def set_xaxis_properties(self):
        self.set_axis_to_time(label=None)

    def draw(
        self, data: xarray.DataArray, x_name="time", y_name="lat", fig=None, ax=None
    ):
        super().draw(data, x_name=x_name, y_name=y_name)
        self.ax_.set_aspect("auto")


class plot_plev_time(BaseContourPlot):
    def __init__(self, proj=None, log_plevs=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj = proj
        self.log_plevs = log_plevs

    def set_yaxis_properties(self):
        self.set_axis_to_plev()

    def set_xaxis_properties(self):
        self.set_axis_to_time(label=None)

    def draw(
        self, data: xarray.DataArray, x_name="time", y_name="plev", fig=None, ax=None
    ):
        super().draw(data, x_name=x_name, y_name=y_name)
        self.ax_.set_aspect("auto")


# class plot_time_plev(plot_lat_plev):
#     def __init__(self, log_plevs=False, panel=(0.1, 0.12, 0.75, 0.8), *args, **kwargs):
#         super().__init__(panel=panel, log_plevs=log_plevs, *args, **kwargs)

#     def _get_coords(
#         self, data: xarray.DataArray, fig=None, x_name="time", y_name="plev"
#     ):
#         # data, x, y = data, data[x_name], data[y_name]
#         x = data.indexes[x_name].to_datetimeindex(unsafe=True)
#         y = data[y_name].values
#         return x, y

#     def draw(self, data: xarray.DataArray, fig=None) -> None:

#         self.lat_, self.plev_ = self._get_coords(data)
#         var = np.ma.squeeze(data.values)

#         if self.print_stats:
#             self.stats_ = [var.min(), var.mean(), var.max()]

#         # Contour levels
#         self.levels_ = var.min() + (var.max() - var.min()) * np.linspace(
#             0, 1, self.nlevels
#         )
#         norm = colors.BoundaryNorm(boundaries=self.levels_, ncolors=256)

#         if fig is None:
#             """Create figure if none is given"""
#             fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
#         # Add a map projection
#         ax = fig.add_axes(self.panel, projection=self.proj)

#         # get cmap
#         self.cmap_ = get_colormap(self.cmap_name)

#         # plot contour map
#         p1 = ax.contourf(
#             self.lat_,
#             self.plev_,
#             var,
#             transform=self.proj,
#             # norm=norm,
#             levels=self.levels_,
#             cmap=self.cmap_,
#             extend="both",
#         )

#         # Full world would be aspect 360/(2*180) = 1
#         # ax.set_aspect((lon_east - lon_west) / (2 * (lat_north - lat_south)))
#         # ax.set_aspect("auto")
#         # ax.coastlines(lw=0.35)
#         self.ax_, self.fig_ = ax, fig
#         self.contour_plot_ = p1

#     def finish(self, plotTitle={"fontsize": 14.0}, plotSideTitle={"fontsize": 7.5}):
#         ax, fig = self.ax_, self.fig_
#         lat, plev = self.lat_, self.plev_

#         # Add title
#         if self.lhs_title is not None:
#             ax.set_title(self.lhs_title, loc="left", fontdict=plotSideTitle)
#         if self.title is not None:
#             ax.set_title(self.title, fontdict=plotTitle)
#         if self.rhs_title is not None:
#             ax.set_title(self.rhs_title, loc="right", fontdict=plotSideTitle)

#         # plev_ticks = np.array([i * 20000 for i in range(6)])
#         # self.ax_.set_yticks(plev_ticks)
#         # self.ax_.set_ylabel("pressure (hPa)")
#         # if self.log_plevs:
#         #     ax.set_yscale("log")

#         self.ax_.set_xlabel("year")
#         self.ax_.set_yscale("log")
#         self.ax_.invert_yaxis()

#         # add stats on top of colorbar
#         # colorbar panel for reference:
#         #   colorbar_rect=(0.89, 0.12, 0.0326, 0.66)
#         if self.print_stats:
#             # Min, Mean, Max
#             fig.text(
#                 0.88,
#                 panel[3] + 0.08,
#                 "Max\nMean\nMin",
#                 ha="left",
#                 fontdict=plotSideTitle,
#             )
#             fig.text(
#                 0.97,
#                 panel[3] + 0.08,
#                 "{0:.2f}\n{1:.2f}\n{2:.2f}".format(*self.stats_),
#                 ha="right",
#                 fontdict=plotSideTitle,
#             )

#         return None
