import numpy as np
import scipy.special as sc

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


def bessel_q(phi, r, Ld, n=3, Lr=1):
    '''q arrising from bessel (Ld = inf)'''
    r = r/Lr  # Scale r
    a = sc.jn_zeros(n, 1)[0]
    q = (((a**2)/4) * np.sin(n*phi)) \
        * (sc.jn(n-2, a*r) - 2*sc.jn(n, a*r) + sc.jn(n+2, a*r)) \
        + ((a/(2*r)) * np.sin(n*phi)) * (sc.jn(n-1, a*r) - sc.jn(n+1, a*r)) \
        - (((n**2)/(r**2)) * np.sin(n*phi)) * sc.jn(n, a*r)
    return q/Lr**2


def bessel(phi, r, n, Lr=1):
    '''Return Bessel values'''
    r = r/Lr  # Scale r
    a = sc.jn_zeros(n, 1)[0]
    z = np.sin(n*phi) * sc.jn(n, a*r)
    return z