SCHWARZSCHILD_CASES =[
    {"name": "schwarzschild_b2", "b_inf": 2.0},
    {"name": "schwarzschild_b3", "b_inf": 3.0},
    {"name": "schwarzschild_b4", "b_inf": 4.0},
    {"name": "schwarzschild_b5", "b_inf": 5.0},
]

import numpy as np

P0 = 6.0
B_INF = 3.0

KERR_NEWMAN_CASES = [
    {
        "panel": "a",
        "name": "extremal_kerr_corot",
        "a": 1.0,
        "rho_Q": 0.0,
        "ell_sign": +1,
        "b_inf": B_INF,
        "P_min": 1.4,
        "P_max": P0,
        "n_annuli": 21,
    },
    {
        "panel": "b",
        "name": "extremal_rn",
        "a": 0.0,
        "rho_Q": 1.0,
        "ell_sign": +1,
        "b_inf": B_INF,
        "P_min": 1.0,
        "P_max": P0,
        "n_annuli": 21,
    },
    {
        "panel": "c",
        "name": "kn_corot",
        "a": 2.0 / 5.0,
        "rho_Q": 4.0 / 5.0,
        "ell_sign": +1,
        "b_inf": B_INF,
        "P_min": 1.0 + np.sqrt(1.0 / 5.0),
        "P_max": P0,
        "n_annuli": 21,
    },
    {
        "panel": "d",
        "name": "kn_counter",
        "a": -2.0 / 5.0,
        "rho_Q": 4.0 / 5.0,
        "ell_sign": +1,
        "b_inf": B_INF,
        "P_min": 1.96,
        "P_max": P0,
        "n_annuli": 21,
    }
]