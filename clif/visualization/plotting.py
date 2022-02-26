from email.mime import base
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import xarray

from .utils import add_cyclic, get_colormap, determine_tick_step, get_ax_size

try:
    from cartopy.util import add_cyclic_point
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
except:
    print("To access the plotting functions, you must have cartopy installed.")

# (left, bottom, width, height) for plotting map
panel = (0.1, 0.1, 0.75, 0.8)

# range of lat and lon coords
lon_west, lon_east, lat_south, lat_north = (
    0,
    360,
    -90,
    90,
)


class baseplot:
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

    def draw(self, data: xarray.DataArray, **draw_params):
        pass

    def add_colorbar(self, **colorbar_params):
        pass

    def finish(self, **finish_params):
        pass

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


class plot_lat_lon(baseplot):
    def __init__(
        self,
        proj=ccrs.PlateCarree(central_longitude=180),
        region="global",
        *args,
        **kwargs
    ):
        super().__init__(proj=proj, region=region, *args, **kwargs)

    def _get_coords(self, data: xarray.DataArray, fig=None, x_name="lat", y_name="lon"):
        data_ma, x, y = add_cyclic(data, lat_name=x_name, lon_name=y_name)
        y = np.array(y)
        x = x.values  # var.getLatitude()
        return x, y, data_ma

    def draw(
        self, data: xarray.DataArray, fig=None, lat_name="lat", lon_name="lon"
    ) -> None:

        self.lat_, self.lon_, data_ma = self._get_coords(data)
        # data, lat, lon = add_cyclic(data, lat_name=lat_name, lon_name=lon_name)
        lat, lon, data_ma = self.lat_, self.lon_, data_ma
        var = np.ma.squeeze(data_ma)

        if self.print_stats:
            self.stats_ = [var.min(), var.mean(), var.max()]

        # Contour levels
        self.levels_ = var.min() + (var.max() - var.min()) * np.linspace(
            0, 1, self.nlevels
        )
        # norm = colors.BoundaryNorm(boundaries=levels, ncolors=256)

        if fig is None:
            """Create figure if none is given"""
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        # Add a map projection
        ax = fig.add_axes(self.panel, projection=self.proj)

        # get cmap
        self.cmap_ = get_colormap(self.cmap_name)

        # plot contour map
        p1 = ax.contourf(
            lon,
            lat,
            var,
            transform=self.proj,
            norm=None,
            levels=self.levels_,
            cmap=self.cmap_,
            extend="both",
        )

        # Full world would be aspect 360/(2*180) = 1
        ax.set_aspect((lon_east - lon_west) / (2 * (lat_north - lat_south)))
        ax.coastlines(lw=0.35)
        self.ax_, self.fig_ = ax, fig
        self.contour_plot_ = p1

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
        # Color bar
        fig = self.fig_
        cbax = fig.add_axes(rect)
        cbar = fig.colorbar(self.contour_plot_, cax=cbax, drawedges=True, alpha=0.5)
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

    def finish(self, plotTitle={"fontsize": 14.0}, plotSideTitle={"fontsize": 7.5}):
        ax, fig = self.ax_, self.fig_
        lat, lon = self.lat_, self.lon_

        # Add title

        if self.lhs_title is not None:
            ax.set_title(self.lhs_title, loc="left", fontdict=plotSideTitle)
        if self.title is not None:
            ax.set_title(self.title, fontdict=plotTitle)
        if self.rhs_title is not None:
            ax.set_title(self.rhs_title, loc="right", fontdict=plotSideTitle)

        # add axis labels
        lon_covered = lon_east - lon_west
        lon_step = determine_tick_step(lon_covered)
        xticks = np.arange(lon_west, lon_east, lon_step)
        # Subtract 0.50 to get 0 W to show up on the right side of the plot.
        # If less than 0.50 is subtracted, then 0 W will overlap 0 E on the left side of the plot.
        # If a number is added, then the value won't show up at all.
        xticks = np.append(xticks, lon_east - 0.50)
        proj = ccrs.PlateCarree(central_longitude=180)
        lat_covered = lat_north - lat_south
        lat_step = determine_tick_step(lat_covered)
        yticks = np.arange(lat_south, lat_north, lat_step)
        yticks = np.append(yticks, lat_north)
        if not self.show_full:
            yticks[0], yticks[-1] = np.ceil(lat.min()), np.ceil(lat.max())

        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(
            zero_direction_label=True, number_format=".0f"
        )
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(labelsize=8.0, direction="out", width=1)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")

        # add stats on top of colorbar
        # colorbar panel for reference:
        #   colorbar_rect=(0.89, 0.12, 0.0326, 0.66)
        if self.print_stats:
            # Min, Mean, Max
            fig.text(
                0.88,
                panel[3] + 0.08,
                "Max\nMean\nMin",
                ha="left",
                fontdict=plotSideTitle,
            )
            fig.text(
                0.97,
                panel[3] + 0.08,
                "{0:.2f}\n{1:.2f}\n{2:.2f}".format(*self.stats_),
                ha="right",
                fontdict=plotSideTitle,
            )

        return None


class plot_lat_plev(baseplot):
    """Plot the lat/lon field using cartopy

    Plot an xarray DataArray lat/ lon field
    """

    def __init__(self, proj=None, log_plevs=False, *args, **kwargs):
        super().__init__(proj=proj, *args, **kwargs)
        self.log_plevs = log_plevs

    def _get_coords(
        self, data: xarray.DataArray, fig=None, x_name="lat", y_name="plev"
    ):
        data, x, y = data, data[x_name], data[y_name]
        y = y.values
        x = x.values  # var.getLatitude()
        return x, y

    def draw(self, data: xarray.DataArray, fig=None) -> None:

        self.lat_, self.plev_ = self._get_coords(data)
        var = np.ma.squeeze(data.values)

        if self.print_stats:
            self.stats_ = [var.min(), var.mean(), var.max()]

        # Contour levels
        self.levels_ = var.min() + (var.max() - var.min()) * np.linspace(
            0, 1, self.nlevels
        )
        norm = colors.BoundaryNorm(boundaries=self.levels_, ncolors=256)

        if fig is None:
            """Create figure if none is given"""
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        # Add a map projection
        ax = fig.add_axes(self.panel, projection=self.proj)

        # get cmap
        self.cmap_ = get_colormap(self.cmap_name)

        # plot contour map
        p1 = ax.contourf(
            self.lat_,
            self.plev_,
            var,
            transform=self.proj,
            # norm=norm,
            levels=self.levels_,
            cmap=self.cmap_,
            extend="both",
        )

        # Full world would be aspect 360/(2*180) = 1
        # ax.set_aspect((lon_east - lon_west) / (2 * (lat_north - lat_south)))
        ax.set_aspect("auto")
        # ax.coastlines(lw=0.35)
        self.ax_, self.fig_ = ax, fig
        self.contour_plot_ = p1

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
        # Color bar
        fig = self.fig_
        cbax = fig.add_axes(rect)
        cbar = fig.colorbar(self.contour_plot_, cax=cbax, drawedges=True, alpha=0.5)
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
        self.ax_cbar_ = cbax

    def finish(self, plotTitle={"fontsize": 14.0}, plotSideTitle={"fontsize": 7.5}):
        ax, fig = self.ax_, self.fig_
        lat, plev = self.lat_, self.plev_

        # Add title
        if self.lhs_title is not None:
            ax.set_title(self.lhs_title, loc="left", fontdict=plotSideTitle)
        if self.title is not None:
            ax.set_title(self.title, fontdict=plotTitle)
        if self.rhs_title is not None:
            ax.set_title(self.rhs_title, loc="right", fontdict=plotSideTitle)

        ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])  # , crs=ccrs.PlateCarree())
        ax.set_xlim(-90, 90)
        lat_formatter = LatitudeFormatter()
        # ax.xaxis.set_major_formatter(lon_formatter)
        ax.xaxis.set_major_formatter(lat_formatter)
        ax.tick_params(labelsize=8.0, direction="out", width=1)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        if self.log_plevs:
            ax.set_yscale("log")
        plev = self.plev_

        # plev_ticks = plev_ticks[::-1]
        plev_ticks = np.array([i * 20000 for i in range(6)])[::-1]
        self.ax_.set_yticks(plev_ticks)

        self.ax_.set_ylabel("pressure (hPa)")
        # ax.set_yscale("log")
        self.ax_.invert_yaxis()

        # add stats on top of colorbar
        # colorbar panel for reference:
        #   colorbar_rect=(0.89, 0.12, 0.0326, 0.66)
        if self.print_stats:
            # Min, Mean, Max
            fig.text(
                0.88,
                panel[3] + 0.08,
                "Max\nMean\nMin",
                ha="left",
                fontdict=plotSideTitle,
            )
            fig.text(
                0.97,
                panel[3] + 0.08,
                "{0:.2f}\n{1:.2f}\n{2:.2f}".format(*self.stats_),
                ha="right",
                fontdict=plotSideTitle,
            )

        return None


class plot_lat_time(plot_lat_plev):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_coords(
        self, data: xarray.DataArray, fig=None, x_name="lat", y_name="time"
    ):
        # data, x, y = data, data[x_name], data[y_name]
        y = data.indexes[y_name].to_datetimeindex(unsafe=True)
        x = data[x_name].values
        return x, y

    def finish(self, plotTitle={"fontsize": 14.0}, plotSideTitle={"fontsize": 7.5}):
        ax, fig = self.ax_, self.fig_
        lat, plev = self.lat_, self.plev_

        # Add title
        if self.lhs_title is not None:
            ax.set_title(self.lhs_title, loc="left", fontdict=plotSideTitle)
        if self.title is not None:
            ax.set_title(self.title, fontdict=plotTitle)
        if self.rhs_title is not None:
            ax.set_title(self.rhs_title, loc="right", fontdict=plotSideTitle)

        ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])  # , crs=ccrs.PlateCarree())
        ax.set_xlim(-90, 90)
        lat_formatter = LatitudeFormatter()
        # ax.xaxis.set_major_formatter(lon_formatter)
        ax.xaxis.set_major_formatter(lat_formatter)
        ax.tick_params(labelsize=8.0, direction="out", width=1)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        # if self.log_plevs:
        #     ax.set_yscale("log")
        # plev = self.plev_

        # # plev_ticks = plev_ticks[::-1]
        # plev_ticks = np.array([i * 20000 for i in range(6)])
        # self.ax_.set_yticks(plev_ticks)

        self.ax_.set_ylabel("year")
        # # ax.set_yscale("log")
        # self.ax_.invert_yaxis()

        # add stats on top of colorbar
        # colorbar panel for reference:
        #   colorbar_rect=(0.89, 0.12, 0.0326, 0.66)
        if self.print_stats:
            # Min, Mean, Max
            fig.text(
                0.88,
                panel[3] + 0.08,
                "Max\nMean\nMin",
                ha="left",
                fontdict=plotSideTitle,
            )
            fig.text(
                0.97,
                panel[3] + 0.08,
                "{0:.2f}\n{1:.2f}\n{2:.2f}".format(*self.stats_),
                ha="right",
                fontdict=plotSideTitle,
            )

        return None


class plot_time_plev(plot_lat_plev):
    def __init__(self, log_plevs=False, panel=(0.1, 0.12, 0.75, 0.8), *args, **kwargs):
        super().__init__(panel=panel, log_plevs=log_plevs, *args, **kwargs)

    def _get_coords(
        self, data: xarray.DataArray, fig=None, x_name="time", y_name="plev"
    ):
        # data, x, y = data, data[x_name], data[y_name]
        x = data.indexes[x_name].to_datetimeindex(unsafe=True)
        y = data[y_name].values
        return x, y

    def draw(self, data: xarray.DataArray, fig=None) -> None:

        self.lat_, self.plev_ = self._get_coords(data)
        var = np.ma.squeeze(data.values)

        if self.print_stats:
            self.stats_ = [var.min(), var.mean(), var.max()]

        # Contour levels
        self.levels_ = var.min() + (var.max() - var.min()) * np.linspace(
            0, 1, self.nlevels
        )
        norm = colors.BoundaryNorm(boundaries=self.levels_, ncolors=256)

        if fig is None:
            """Create figure if none is given"""
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        # Add a map projection
        ax = fig.add_axes(self.panel, projection=self.proj)

        # get cmap
        self.cmap_ = get_colormap(self.cmap_name)

        # plot contour map
        p1 = ax.contourf(
            self.lat_,
            self.plev_,
            var,
            transform=self.proj,
            # norm=norm,
            levels=self.levels_,
            cmap=self.cmap_,
            extend="both",
        )

        # Full world would be aspect 360/(2*180) = 1
        # ax.set_aspect((lon_east - lon_west) / (2 * (lat_north - lat_south)))
        # ax.set_aspect("auto")
        # ax.coastlines(lw=0.35)
        self.ax_, self.fig_ = ax, fig
        self.contour_plot_ = p1

    def finish(self, plotTitle={"fontsize": 14.0}, plotSideTitle={"fontsize": 7.5}):
        ax, fig = self.ax_, self.fig_
        lat, plev = self.lat_, self.plev_

        # Add title
        if self.lhs_title is not None:
            ax.set_title(self.lhs_title, loc="left", fontdict=plotSideTitle)
        if self.title is not None:
            ax.set_title(self.title, fontdict=plotTitle)
        if self.rhs_title is not None:
            ax.set_title(self.rhs_title, loc="right", fontdict=plotSideTitle)

        # plev_ticks = np.array([i * 20000 for i in range(6)])
        # self.ax_.set_yticks(plev_ticks)
        self.ax_.set_ylabel("pressure (hPa)")
        # if self.log_plevs:
        #     ax.set_yscale("log")

        self.ax_.set_xlabel("year")
        # if self.log_plevs:
        #     self.ax_.set_yscale("log")
        self.ax_.invert_yaxis()

        # add stats on top of colorbar
        # colorbar panel for reference:
        #   colorbar_rect=(0.89, 0.12, 0.0326, 0.66)
        if self.print_stats:
            # Min, Mean, Max
            fig.text(
                0.88,
                panel[3] + 0.08,
                "Max\nMean\nMin",
                ha="left",
                fontdict=plotSideTitle,
            )
            fig.text(
                0.97,
                panel[3] + 0.08,
                "{0:.2f}\n{1:.2f}\n{2:.2f}".format(*self.stats_),
                ha="right",
                fontdict=plotSideTitle,
            )

        return None
