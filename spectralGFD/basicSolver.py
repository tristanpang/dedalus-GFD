from spectralGFD import *
import numpy as np
import dedalus.public as d3
from functools import partial
import time
import matplotlib.pyplot as plt
import copy

class DedalusSolver:
    '''
    Basic class for solver.

    IMPORTANT: implement make_problem and
    solve_problem in subclass.

    Useful methods:
    run = make_space + make_problem + solve_problem
          (__init__ executes run once)
    actual = set actual func
    compute_error = graph overview of error for saved run
    error_lists + error_plots = graph error for varying N

    Other methods:
    plot
    plot_result
    plot_actual
    plot_gridpoints
    '''
    def __init__(self, Nphi, Nr, Lr=1, *, dealias=1):
        # Export vars
        self.Nphi = Nphi
        self.Nr = Nr
        self.Lr = Lr
        self.dealias = dealias

        # Run
        self.make_space()

    def make_space(self):
        '''Setup Dedalus basis and field'''
        # Import vars
        Nphi = self.Nphi
        Nr = self. Nr
        Lr = self.Lr
        dealias = self.dealias

        # Parameters
        dtype = np.float64

        # Bases
        coords = d3.PolarCoordinates('phi', 'r')
        dist = d3.Distributor(coords, dtype=dtype)
        disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=Lr,
                            dealias=dealias, dtype=dtype)  # Circular domain
        edge = disk.edge
        phi, r = dist.local_grids(disk)

        # Field
        u = dist.Field(name='u', bases=disk)

        # Export vars
        self.coords = coords
        self.dist = dist
        self.disk = disk
        self.edge = edge
        self.u = u
        self.phi = phi
        self.r = r

    def run(self, local=True, save_every=None, save_name=None):
        '''Fully run problem from scratch'''
        self.make_space()
        self.make_problem()

        time_0 = time.perf_counter()  # Start timer
        self.solve_problem(local=local, save_every=save_every, save_name=save_name)
        time_tot = time.perf_counter() - time_0

        self.time = time_tot

    def actual(self, actual_func):
        '''Return array of values of actual function on
        input of actual_func = type func'''
        # Import vars
        phi = self.phi
        r = self.r
        Lr = self.Lr

        self.actual_func = actual_func
        scaled_func = partial(actual_func, Lr=Lr)
        return func_on_mesh(scaled_func, phi, r)

    def plot(self, z, ax=None, filename=None, title=None, cax=None):
        '''Generic plot via polar_plot. Input z'''
        # Import vars
        phi = self.phi
        r = self.r
        polar_plot(phi, r, z, ax=ax, filename=filename, title=title, cax=cax)

    def plot_result(self, ax=None, filename=None, title=None, cax=None):
        '''Plot of results using polar_plot'''
        z = self.ug.T
        self.plot(z, ax, filename, title, cax)

    def plot_actual(self, ax=None, filename=None, title=None, cax=None):
        '''Plot of actual using polar_plot'''
        z = self.actual(self.actual_func)
        self.plot(z, ax, filename, title, cax)

    def plot_gridpoints(self, alpha=0.5):
        '''Plot polar gridpoints given radius and azimuth'''
        # Import vars
        Nphi = self.Nphi
        Nr = self.Nr
        coords = self.coords
        dist = self.dist
        disk = self.disk
        phi, r = dist.local_grids(disk)

        phi_mesh, r_mesh = self.mesh()
        x_vals = (r_mesh * np.cos(phi_mesh)).ravel()
        y_vals = (r_mesh * np.sin(phi_mesh)).ravel()

        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        ax.set_aspect('equal')

        plt.title(f'Grid points: Nphi={Nphi}, Nr={Nr}')
        plt.scatter(x_vals, y_vals, marker='x', s=10, alpha=alpha)
        plt.show()

    def integral_error(self):
        '''Compute sqrt(int(computed-actual)^2)'''
        # Import vars
        disk = self.disk
        actual_func = self.actual_func
        phi = self.phi
        r = self.r
        u = self.u  # field
        dist = self.dist

        # Actual field
        actual = dist.Field(name='actual', bases=disk)
        actual['g'] = actual_func(phi, r)

        # Compute errors
        error = np.sqrt(d3.integ((u - actual)**2)).evaluate()['g']
        return error[0, 0]

    def naive_error(self):
        '''Sum of errors over grid'''
        errors = self.actual - self.ug.T
        norm = np.linalg.norm(errors, 2)  # 2-norm
        weighted_norm = norm * 1/np.sqrt(self.Nr * self.Nphi)  # weighted 2-norm
        return weighted_norm

    def compute_error(self, make_plots=True, filename=None):
        '''Plots and computes error of given functions by
        comparing with actual_func on mesh. Assumes problem
        is presolved.'''
        # Import vars
        Nr = self.Nr
        Nphi = self.Nphi
        ug = self.ug
        actual = self.actual(self.actual_func)

        errors = actual - ug.T
        weighted_norm = self.integral_error()

        if make_plots is True:
            print('Error:', weighted_norm)
            # Plotting
            fig, axs = plt.subplots(1, 3, figsize=(20, 6))

            for ax in axs:
                ax.ticklabel_format(style='sci')

            self.plot_result(ax=axs[0], title='Dedalus')
            self.plot_actual(ax=axs[1], title='Actual')
            self.plot(errors, ax=axs[2], title='Error')
            plt.tight_layout()
            plt.show()

            if filename is not None:
                fig.savefig(filename+'.png', dpi=fig.dpi,
                            bbox_inches='tight')

        # Export vars
        self.error = weighted_norm

    def error_lists(self, Nphi_list, Nr_list):
        '''Solve problem with various different Nphi and Nr
        (input as numpy lists, nb needs Nphi = 0 mod 4).
        Outputs lists of errors and times.'''
        # Import vars
        Nphi = self.Nphi
        Nr = self.Nr
        loop_solver = copy.copy(self)

        # Loop Nr
        loop_solver.Nphi = Nphi
        Nr_error_list = []
        Nr_times_list = []

        for Nr_loop in Nr_list:
            print('Nr =', Nr_loop)

            loop_solver.Nr = Nr_loop
            loop_solver.run()
            loop_solver.compute_error(make_plots=False)

            Nr_error_list.append(loop_solver.error)
            Nr_times_list.append(loop_solver.time)

        # Loop Nphi
        loop_solver.Nr = Nr
        Nphi_error_list = []
        Nphi_times_list = []

        for Nphi_loop in Nphi_list:
            print('Nphi =', Nphi_loop)

            loop_solver.Nphi = Nphi_loop
            loop_solver.run()
            loop_solver.compute_error(make_plots=False)

            Nphi_error_list.append(loop_solver.error)
            Nphi_times_list.append(loop_solver.time)

        print('Done!')

        # Export vars
        self.Nr_list = Nr_list
        self.Nphi_list = Nphi_list
        self.Nr_error_list = Nr_error_list
        self.Nr_times_list = Nr_times_list
        self.Nphi_error_list = Nphi_error_list
        self.Nphi_times_list = Nphi_times_list

    def error_plots(self, truncs=[20, 20]):
        '''Plots error and time lists from self.error_lists
           trunc = point to truncate best fit'''
        # Import vars
        Nr_list = self.Nr_list
        Nphi_list = self.Nphi_list
        Nr_error_list = self.Nr_error_list
        Nr_times_list = self.Nr_times_list
        Nphi_error_list = self.Nphi_error_list
        Nphi_times_list = self.Nphi_times_list

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        ax = axs[0]
        ax.plot(Nr_list, Nr_error_list, '-o')
        ax.set(xlabel="Number of basis functions $N_r$",
               ylabel="Normed errors")
        ax.set_yscale('log')
        ax.set_title(f"Varying $N_r$, $N_\phi={self.Nphi}$")

        trunc = truncs[0]
        coeffs = np.polyfit(Nr_list[:trunc], np.log(Nr_error_list[:trunc]), 1)
        poly = str(np.poly1d(coeffs, variable=r'$N_r$'))[2:]
        ax.plot(Nr_list[:trunc], np.exp(np.poly1d(coeffs)(Nr_list[:trunc])),
                label=f'exp({poly})')
        ax.legend()

        ax = axs[1]
        ax.plot(Nphi_list, Nphi_error_list, '-o')
        ax.set(xlabel=r"Number of basis functions $N_\phi$",
               ylabel="Normed errors")
        ax.set_yscale('log')
        ax.set_title(f"Varying $N_\phi$, $N_r={self.Nr}$")

        trunc = truncs[1]
        if trunc is not None:
            coeffs = np.polyfit(Nphi_list[:trunc],
                                np.log(Nphi_error_list[:trunc]), 1)
            poly = str(np.poly1d(coeffs, variable=r'$N_\phi$'))[2:]
            ax.plot(Nphi_list[:trunc],
                    np.exp(np.poly1d(coeffs)(Nphi_list[:trunc])),
                    label=f'exp({poly})')
            ax.legend()

        fig.tight_layout()
        plt.show()

    def error_plots_times(self):
        '''Plots error and time lists from self.error_lists'''
        # Import vars
        Nr_list = self.Nr_list
        Nphi_list = self.Nphi_list
        Nr_error_list = self.Nr_error_list
        Nr_times_list = self.Nr_times_list
        Nphi_error_list = self.Nphi_error_list
        Nphi_times_list = self.Nphi_times_list

        # Plot r
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        ax = axs[0]
        ax.plot(Nr_list, Nr_error_list, '-o')
        ax.set(xlabel="Nr", ylabel="Normed errors")
        ax.set_yscale('log')

        ax = axs[1]
        ax.plot(Nr_list, Nr_times_list)
        ax.set(xlabel="Nr", ylabel="Time")
        ax.set_yscale('log')
        plt.suptitle('Errors and Times vs varying radial basis size')

        fig.tight_layout()
        plt.show()

        # Plot phi
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        ax = axs[0]
        ax.plot(Nphi_list, Nphi_error_list, '-o')
        ax.set(xlabel="Nphi", ylabel="Normed errors")
        ax.set_yscale('log')

        ax = axs[1]
        ax.plot(Nphi_list, Nphi_times_list)
        ax.set(xlabel="Nphi", ylabel="Time")
        ax.set_yscale('log')
        plt.suptitle('Errors and Times vs varying azimuthal basis size')

        fig.tight_layout()
        plt.show()

    def mesh(self):
        '''Mesh phi, r'''
        phi_mesh, r_mesh = np.meshgrid(self.phi, self.r)

        # Export vars
        self.phi_mesh = phi_mesh
        self.r_mesh = r_mesh
        return phi_mesh, r_mesh

    def to_cartesian(self):
        '''Polar to cartesian'''
        phi_mesh, r_mesh = self.mesh()
        x = r_mesh * np.cos(phi_mesh)
        y = r_mesh * np.sin(phi_mesh)
        return x, y

    def to_square(self):
        '''Map to square'''
        u, v = self.to_cartesian()
        x, y = circle_to_square(u, v)
        return x, y