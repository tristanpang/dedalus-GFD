from spectralGFD import *

def gaussian_q(phi, r, Ld, Lr=1):
    '''q arrising from gaussian with Ld'''
    r = r/Lr  # Scale r
    q = 1/Lr**2 * (r**2 - 2) * np.exp(-(r**2)/2) \
        - (1/(Ld**2)) * (np.exp(-(r**2)/2) - np.exp(-1/2))
    return q


def gaussian(phi, r, Lr=1):
    '''Return gaussian values given mesh phi, r'''
    r = r/Lr  # Scale r
    return np.exp(-(r**2)/2) - np.exp(-1/2)


# Params
Ld = 1  # coriolis effect
Nphi, Nr = 2**5, 2**7
Lr = 1

gaussian_lap = Lap_Cor(Nphi, Nr, gaussian_q, Ld=Ld, Lr=Lr)
gaussian_lap.actual(gaussian)
gaussian_lap.compute_error()