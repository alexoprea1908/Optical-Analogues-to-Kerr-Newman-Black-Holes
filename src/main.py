from plots_profiles import make_schwarzschild_profiles, make_kerr_newman_profiles
from plots_ray_tracing import make_figure_5, make_figure_6, make_figure_6_full
from plots_fdfd import make_figure_3, make_figure_4


def main():
    # Figs 1 & 2: Refractive index profiles
    make_schwarzschild_profiles()
    make_kerr_newman_profiles()
    print("Saved profile plots in results/profiles/")

    # Fig 3: FDFD Schwarzschild simulations (b_inf = 2, 3, 4, 5)
    make_figure_3()
    print("Saved Schwarzschild FDFD plots in results/fdfd/")

    #Fig 4: FDFD Kerr-Newman simulations
    make_figure_4()
    print("Saved Kerr-Newman FDFD plots in results/fdfd/")

    # Fig 5: Ray tracing - annulus number impact
    make_figure_5()
    print("Saved annulus number analysis in results/ray_tracing/")

    # Fig 6: Ray tracing - error analysis (Dn and DB0)
    make_figure_6()
    make_figure_6_full()
    print("Saved error analysis plots in results/ray_tracing/")

    print("\nAll figures saved in results/")


if __name__ == "__main__":
    main()