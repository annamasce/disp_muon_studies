import numpy as np

def delta_r2(v1, v2):
    '''Calculates deltaR squared between two particles v1, v2 whose
    eta and phi methods return arrays
    '''
    dphi = (v1.phi - v2.phi + np.pi) % (2 * np.pi) - np.pi
    deta = v1.eta - v2.eta
    dr2 = dphi ** 2 + deta ** 2
    return dr2


def delta_r(v1, v2):
    '''Calculates deltaR between two particles v1, v2 whose
    eta and phi methods return arrays.

    Note: Prefer delta_r2 for cuts.
    '''
    return np.sqrt(delta_r2(v1, v2))

def delta_phi(v1, v2):
    '''Calculates deltaPhi between two particles v1, v2 whose
    phi methods return arrays.
    '''
    dphi = (v1.phi - v2.phi + np.pi) % (2 * np.pi) - np.pi
    return dphi

def sum_pt(v1, v2):
    '''Calculate 2-particle pt for v1 and v2 whose pt, px and py method return arrays.
    '''
    sum_pt_2 = v1.pt**2 + v2.pt**2 + 2 * v1.pt * v2.pt * np.cos(v1.phi - v2.phi)
    sum_pt = np.sqrt(sum_pt_2)
    return sum_pt