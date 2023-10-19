from spectralGFD import *

# Params
Nphi, Nr = 2**7, 2**7  # Grid spacing
n = 3  # initial bessel
time_step = np.pi/400  # Time step
stop_sim_time = 3*np.pi + time_step  # Simulation length
Lr = 1
timestepper = d3.SBDF3
save_every = 200
scales = 2


def initial_func(phi, r):
    return bessel_q(phi, r, Ld=np.inf, n=n, Lr=Lr)


rotation_bessel = rotation_PDE(Nphi, Nr, initial_func, Lr=Lr,
                               stop_sim_time=stop_sim_time, timestep=time_step,
                               timestepper=timestepper,
                               local=False, save_every=save_every, scales=scales)
rotation_bessel.time_plot()