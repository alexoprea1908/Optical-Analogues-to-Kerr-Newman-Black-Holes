import os
import numpy as np
import matplotlib.pyplot as plt

from constants import P0, DEFAULT_NUM_POINTS
from cases import SCHWARZSCHILD_CASES, KERR_NEWMAN_CASES
from Schwarzchild import refractive_index_schwarzschild
from Kerr_Newman import refractive_index_kn_continuous
from annuli import annulus_edges_with_half_ends, sample_piecewise_constant

def make_schwarzschild_profiles():
    # In the paper, the optical black holes use outer radius P0 = 6
    # and different b_inf values for Schwarzschild.
    P = np.linspace(2.0, P0, DEFAULT_NUM_POINTS)

    plt.figure(figsize=(8, 6))
    for case in SCHWARZSCHILD_CASES:
        n = refractive_index_schwarzschild(P, case["b_inf"], n_at_P0=1.0, P0=P0)
        label = rf"$\hat b_{{\infty}}={case['b_inf']}$"
        plt.plot(P, n, label=label)

    plt.xlabel("P")
    plt.ylabel("n(P)")
    plt.title("Schwarzschild scalar refractive index profiles")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs("results/profiles", exist_ok=True)
    plt.savefig("results/profiles/schwarzschild_profiles.png", dpi=200)
    plt.close()



def _step_data(edges, values):
    x = edges[::-1]
    y = np.r_[values[::-1], values[::-1][-1]]
    return x, y


def make_kerr_newman_profiles():
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.ravel()

    for ax, case in zip(axes, KERR_NEWMAN_CASES):
        edges = annulus_edges_with_half_ends(case["P_min"], case["P_max"], case["n_annuli"])
        _, values = sample_piecewise_constant(
            refractive_index_kn_continuous,
            edges,
            case["a"],
            case["rho_Q"],
            case["b_inf"],
            case["ell_sign"],
            case["P_max"],
            1.0,
        )

        x, y = _step_data(edges, values)
        ax.step(x, y, where="post", linewidth=1.5)

        ax.set_xlim(case["P_min"], case["P_max"])
        ax.set_yscale("log")
        ax.set_xlabel(r"$R/M$")
        ax.set_ylabel(r"$n$")
        ax.text(0.06, 0.88, case["panel"], transform=ax.transAxes, fontsize=14, fontweight="bold")

    fig.suptitle("Kerr–Newman scalar refractive index profiles", fontsize=16)
    fig.tight_layout()

    os.makedirs("results/profiles", exist_ok=True)
    fig.savefig("results/profiles/kerr_newman_profiles.png", dpi=250)
    plt.close(fig)