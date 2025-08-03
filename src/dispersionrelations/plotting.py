import numpy as np
from skimage import measure

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.pyplot import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D


plt.rcParams["figure.figsize"] = [8, 4]
plt.rcParams["figure.dpi"] = 100

plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
plt.rc("text", usetex=True)


def prepare_re_im(
    re_min,
    re_max,
    re_n,
    im_min,
    im_max,
    im_n,
    im_near_zero=1e-2, # relative units
    include_im_zero=True,
):
    z_re = np.linspace(re_min, re_max, re_n)
    z_im = np.linspace(im_min, im_max, im_n)
    if im_near_zero != 0.0:
        z_im = np.concat(
            (
                np.linspace(im_min, -np.abs(im_near_zero * im_max), im_n // 2),
                np.linspace(0, 0, int(include_im_zero)),
                np.linspace(np.abs(im_near_zero * im_max), im_max, im_n // 2),
            )
        )
    return z_re, z_im


def prepare_complex_plot_input(function, z_re, z_im, label):
    plot_input = {
        "function": function,
        "label": label,
        "z_re": z_re,
        "z_im": z_im,
    }

    plot_input["z_RE"], plot_input["z_IM"] = np.meshgrid(z_re, z_im)
    plot_input["z_CP"] = plot_input["z_RE"] + 1j * plot_input["z_IM"]
    plot_input["q_CP"] = function(plot_input["z_CP"])

    return plot_input


def figure_3x2(**figkwargs):
    fig = plt.figure(**figkwargs)

    ax_re = fig.add_subplot(2, 3, 1)
    ax_im = fig.add_subplot(2, 3, 2)
    ax_abs = fig.add_subplot(2, 3, 3)
    ax_re_3d = fig.add_subplot(2, 3, 4, projection="3d", computed_zorder=False)
    ax_im_3d = fig.add_subplot(2, 3, 5, projection="3d", computed_zorder=False)
    ax_abs_3d = fig.add_subplot(2, 3, 6, projection="3d", computed_zorder=False)

    return fig


def plot_complex_ax(ax, plot_input, q_function, plotting_function = Axes.contourf, **pfkwargs):
    return plotting_function(ax, plot_input["z_RE"], plot_input["z_IM"], q_function(plot_input["q_CP"]), **pfkwargs)


def plot_contourf_ax(ax, plot_input, q_function, ncontours=20, centered_around_zero=False, **pfkwargs):
    q_current = q_function(plot_input['q_CP'])
    q_max = np.max(q_current)
    q_min = np.min(q_current)
    if centered_around_zero:
        q_max = max(np.abs(q_max), np.abs(q_min))
        q_min = -q_max
    contour_levels = np.linspace(q_min, q_max, ncontours)
    cf = plot_complex_ax(ax, plot_input, q_function, Axes.contourf, **dict(levels=contour_levels, **pfkwargs))
    plt.gcf().colorbar(cf, ax=ax)


def plot_curve_ax2d_ax3d(ax2d, ax3d, x, y, z, **pfkwargs):
    if ax2d is not None:
        ax2d.plot(x, y, **pfkwargs)

    if ax3d is not None:
        ax3d.plot(x, y, z, **pfkwargs)


def scatter_ax2d_ax3d(ax2d, ax3d, x, y, z, **pfkwargs):
    if ax2d is not None:
        ax2d.scatter(x, y, **pfkwargs)
        
    if ax3d is not None:
        ax3d.scatter(x, y, z, **pfkwargs)


def plot_ray_ax2d_ax3d(ax2d, ax3d, x, y, z, **pfkwargs):
    plot_curve_ax2d_ax3d(ax2d, ax3d, x, y, z, **pfkwargs)
    scatter_ax2d_ax3d(ax2d, ax3d, x[:1], y[:1], z[:1], **pfkwargs)


def plot_contours_ax2d_ax3d(ax2d, ax3d, plot_input, q_function, ncontours=20, centered_around_zero=False, **pfkwargs):
    q_current = q_function(plot_input['q_CP'])
    q_max = np.max(q_current)
    q_min = np.min(q_current)
    if centered_around_zero:
        q_max = max(np.abs(q_max), np.abs(q_min))
        q_min = -q_max
    for isovalue in np.linspace(q_min, q_max, ncontours):
        contours = measure.find_contours(q_function(plot_input["q_CP"]), isovalue)
        for contour in contours:
            x_cont = plot_input['z_re'][0] + (plot_input['z_re'][-1] - plot_input['z_re'][0]) * contour[:,1] / (len(plot_input['z_re']) - 1)
            y_cont = plot_input['z_im'][0] + (plot_input['z_im'][-1] - plot_input['z_im'][0]) * contour[:,0] / (len(plot_input['z_im']) - 1)
            z_cont = q_function(plot_input['function'](x_cont + 1j * y_cont))
            plot_curve_ax2d_ax3d(ax2d, ax3d, x_cont, y_cont, z_cont, **pfkwargs)


per_column_default = {
    0: {
        "q_function": np.real,
        "title": "Real part",
        "ncontours": 20,
        "centered_around_zero": True,
        "pfkwargs": dict(cmap="RdYlGn", norm=colors.CenteredNorm()),
    },
    1: {
        "q_function": np.imag,
        "title": "Imaginary part",
        "ncontours": 20,
        "centered_around_zero": True,
        "pfkwargs": dict(cmap="RdYlGn", norm=colors.CenteredNorm()),
    },
    2: {
        "q_function": np.abs,
        "title": "Absolute value",
        "ncontours": 24,
        "centered_around_zero": True,
        "pfkwargs": dict(cmap="Blues", norm=colors.Normalize(vmin=0)),
    }
}

def complex_plot_contours(plot_input, per_column=None):
    fig = figure_3x2(figsize=(10, 6))

    if per_column is None:
        per_column = per_column_default

    for col in per_column.keys():
        pc = per_column[col]
        fig.axes[col].set_title(pc['title'])
        plot_contourf_ax(fig.axes[col], plot_input, pc["q_function"], ncontours=pc["ncontours"], centered_around_zero=pc["centered_around_zero"], **pc["pfkwargs"])
        plot_complex_ax(fig.axes[3 + col], plot_input, pc["q_function"], Axes3D.plot_surface, **pc["pfkwargs"])
        plot_contours_ax2d_ax3d(fig.axes[col], fig.axes[3 + col], plot_input, pc["q_function"], ncontours=pc["ncontours"], centered_around_zero=pc["centered_around_zero"], c='white', alpha=0.3, lw=1)

    for ax in fig.axes[:3]:
        ax.axhline(0, c='white', alpha=0.1)
        ax.axvline(0, c='white', alpha=0.1)
        ax.set_aspect(1)

    fig.tight_layout()

    return fig