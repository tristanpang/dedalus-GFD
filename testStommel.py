from spectralGFD import *

# Params
Nphi, Nr = 256, 256  # Grid spacing
n = 2  # initial bessel
time_step = 6 * 60  # Time step
timestepper = d3.SBDF3
stop_sim_time = 60 * 60 * 24 * 365 * 3  # Simulation length
save_every = 10 * 24 * 7
Lr = 2e6
dealias = 2
scale = 3

# Constants
constants = {
    'F' : 0.1,
    'H' : 500,
    'r0' : 2e-7,
    'beta' : 2e-11,
    'nu' : 80,
    'rho0' : 1000,
    'Q_shift' : 0.01}

def zeta_init(phi, r):
    phi_mesh, r_mesh = np.meshgrid(phi, r)
    return 1e-16 * partial(bessel, n=n, Lr=Lr)(phi_mesh, r_mesh).T

initial_func = [None, zeta_init, 0]

stommel_bessel = stommel_PDE(Nphi, Nr,
                             initial_func,
                             dealias=dealias,
                             stop_sim_time=stop_sim_time,
                             timestep=time_step,
                             timestepper=timestepper,
                             Lr=Lr,
                             scales=scale,
                             local=False,
                             save_every=save_every,
                             **constants)
stommel_bessel.time_plot()