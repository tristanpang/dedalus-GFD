import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import warnings
from IPython.display import clear_output

def test():
    print('Hello world')


def im_plot(x_vals, y_vals, z, *, ax=None, filename=None, title=None, cax=None, cmap=None):
    ''' Make plot of z(x_vals, y_vals) from input arrays.
    Inputs: phi = azimuth (radians)
            r = radius
            z = function value as array (ensure dimension is phi-by-r)
            ax = ax to plot on
            title = plot title (=filename if title=None)
            cax = colourbar axis (False=disable)
            cmap = custom colour map'''

    if ax is None:
        ax_set = False
        fig = plt.figure(figsize=(5, 5))
        ax = fig.gca()
    else:
        ax_set = True

    if title is not None:
        ax.title.set_text(title)
    else:
        ax.title.set_text(filename)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=("The input coordinates"))
        im = ax.pcolormesh(x_vals, y_vals, z, edgecolors='face', cmap=cmap)

    ax.set_aspect('equal')

    if cax is None:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    elif cax is False:
        pass
    else:
        plt.colorbar(im, cax=cax)

    if ax_set is False:
        if filename is not None:
            fig.savefig(filename+'.png', dpi=fig.dpi, bbox_inches='tight')
        plt.show()
    return im


def polar_plot(phi, r, z, *, ax=None, filename=None, title=None, cax=None, cmap=None):
    ''' Make polar plot of z(phi,r) from input arrays.
    Inputs: phi = azimuth (radians)
            r = radius
            z = function value as array (ensure dimension is phi-by-r)
            ax = ax to plot on
            title = plot title (=filename if title=None)
            cax = colourbar axis (False=disable)
            cmap = custom colour map'''

    phi_mesh, r_mesh = np.meshgrid(phi, r)
    x_vals = r_mesh * np.cos(phi_mesh)
    y_vals = r_mesh * np.sin(phi_mesh)

    im = im_plot(x_vals, y_vals, z, ax=ax,
                 filename=filename, title=title, cax=cax, cmap=cmap)
    return im


def animate(plot_func, t_list, pause=0):
    '''Make animation based on a plot function and time list'''
    for k, t in enumerate(t_list):
        clear_output(wait=True)
        plot_func(k, t)
        plt.pause(pause)
        plt.show()


def func_on_mesh(psi, phi, r):
    '''Return psi(phi, r) on meshed grid'''
    phi_mesh, r_mesh = np.meshgrid(phi, r)
    return psi(phi_mesh, r_mesh)