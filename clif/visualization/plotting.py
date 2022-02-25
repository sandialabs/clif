import numpy as np
import matplotlib.pyplot as plt
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


class plot_lat_lon_field:
    def __init__(
        self,
        cmap_name="e3sm_default",
        proj=ccrs.PlateCarree(central_longitude=180),
        figsize=[8.5, 4.0],
        region="global",
        dpi=150,
        print_stats=True,
        nlevels=30,
        panel=panel,
        title="",
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

    def draw(self, data: xarray.DataArray, fig=None) -> None:

        data, lat, lon = add_cyclic(data)
        lon = np.array(lon)  # var.getLongitude()
        lat = lat.values  # var.getLatitude()
        self.lat_, self.lon_ = lat, lon
        var = np.ma.squeeze(data)

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
        ax.coastlines(lw=0.5)
        self.ax_, self.fig_ = ax, fig
        self.contour_plot_ = p1

    def add_colorbar(self, rect=(0.89, 0.12, 0.0326, 0.66)):
        """draw color bar relative to location of map graph

        rect = (left, bottom, width, height)
        see self.panel for reference

        """
        if not self.print_stats:
            rect = (0.89, 0.12, 0.0326, panel[3] - 0.05)
        # Color bar
        fig = self.fig_
        cbax = fig.add_axes(rect)
        cbar = fig.colorbar(self.contour_plot_, cax=cbax)
        w, h = get_ax_size(fig, cbax)

        if self.levels_ is None:
            cbar.ax.tick_params(labelsize=9.0, length=0)

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
            cbar.ax.tick_params(labelsize=9.0, pad=pad, length=0)

    def finish(self, plotTitle={"fontsize": 11.5}, plotSideTitle={"fontsize": 9.5}):
        ax, fig = self.ax_, self.fig_
        lat, lon = self.lat_, self.lon_

        # Add title
        title_in = self.title
        if isinstance(title_in, str):
            title = [None, title_in, None]
        assert (
            len(title) == 3
        ), "title must be a list of length 3 for left, middle and right"

        if title[0] is not None:
            ax.set_title(title[0], loc="left", fontdict=plotSideTitle)
        if title[1] is not None:
            ax.set_title(title[1], fontdict=plotTitle)
        if title[2] is not None:
            ax.set_title(title[2], loc="right", fontdict=plotSideTitle)

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
                0.87,
                0.22 + 0.6,
                "Max\nMean\nMin",
                ha="left",
                fontdict=plotSideTitle,
            )
            fig.text(
                0.98,
                0.22 + 0.6,
                "{0:.2f}\n{1:.2f}\n{2:.2f}".format(*self.stats_),
                ha="right",
                fontdict=plotSideTitle,
            )

        return None

    def draw_and_show(
        self,
        data: xarray.DataArray,
        fig=None,
        save=False,
        file_name=None,
        colorbar_params={},
        finish_params={},
    ):
        self.draw(data, fig=fig)
        self.add_colorbar(**colorbar_params)
        self.finish(**finish_params)
        if save:
            assert file_name is not None, "Must provide a valid file name"
            self.fig_.savefig(file_name)
        plt.show()
        return self
