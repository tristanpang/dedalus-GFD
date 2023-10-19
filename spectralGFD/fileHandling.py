import h5py
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
from IPython.display import HTML, display
from spectralGFD import *

def animate_file(save_name, file_num=1, variable_name='q', frames=None, cmap=None):
    '''Animate from folder called save_name'''
    with h5py.File(f"saves/{save_name}/{save_name}_s{file_num}.h5",
                   mode='r') as file:
        # Load dataset
        q = file['tasks'][variable_name]
        t_list = q.dims[0][0]
        phi_list = q.dims[1][0]
        r_list = q.dims[2][0]

        if frames is None:
            frames = range(len(t_list))

        def plot_func(k):
            polar_plot(phi_list, r_list, q[k].T, ax=ax[0],
                       title=f'Time: {t_list[k]:.3f}', cax=ax[1], cmap=cmap)

        fig, ax = plt.subplots(1, 3, width_ratios=[10, 1, 1], figsize=(6, 5))
        ax[2].axis('off')
        ani2 = matplotlib.animation.FuncAnimation(fig, plot_func,
                                                  frames=frames)
        plt.show()



def time_plot_file(save_name, file_num=1, plot_t_list=[0, .5, 1],
                   variable_name='q', filename=None):
    '''From folder save_name,
    Plots PDE over time. `plot_t_list` is list
    of normalised time in [0,1] to be plotted.'''
    with h5py.File(f"saves/{save_name}/{save_name}_s{file_num}.h5",
                   mode='r') as file:
        # Load dataset
        q = file['tasks'][variable_name]
        t_list = q.dims[0][0]
        phi_list = q.dims[1][0]
        r_list = q.dims[2][0]

        # Plotting
        plot_length = len(plot_t_list)
        fig, axs = plt.subplots(1, plot_length, figsize=(20, 6))

        for i, t in enumerate(plot_t_list):
            t_ind = int(t * (len(t_list) - 1))
            polar_plot(phi_list, r_list, q[t_ind].T, ax=axs[i],
                       title=f'Time = {t_list[t_ind]:.3}')
        fig.tight_layout()
        plt.show()

        if filename is not None:
            fig.savefig(filename+'.png', dpi=fig.dpi,
                        bbox_inches='tight')


def load_snapshot(save_name, file_num=1, index=0,
                  variable_name='q'):
    '''From folder save_name.'''
    with h5py.File(f"saves/{save_name}/{save_name}_s{file_num}.h5",
                   mode='r') as file:
        # Load dataset
        q = file['tasks'][variable_name]
        t_list = q.dims[0][0]
        phi_list = q.dims[1][0]
        r_list = q.dims[2][0]

        return phi_list[:], r_list[:], q[index], t_list[index]