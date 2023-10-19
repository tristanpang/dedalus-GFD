from spectralGFD import *

## Gaussian

# Params
Ld = 1  # coriolis effect
Nphi, Nr = 2**5, 2**7
Lr = 1

gaussian_lap = Lap_Cor(Nphi, Nr, gaussian_q, Ld=Ld, Lr=Lr)
gaussian_lap.actual(gaussian)
gaussian_lap.compute_error()

## Bessel

# Params
n = 3  # Bessel order
Nphi, Nr = 2**8, 2**8
Lr = 1

bessel_lap = Lap_Cor(Nphi, Nr, partial(bessel_q, n=n), Ld=np.inf, Lr=Lr)
bessel_lap.actual(partial(bessel, n=n))
bessel_lap.compute_error()