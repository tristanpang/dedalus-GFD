from spectralGFD import *

save_name = 'Snapshots 2023-10-19 13:13:16 rotation_PDE'
frames = np.arange(120, step = 10, dtype=int)
v=animate_file(save_name, variable_name='psi', frames=frames)