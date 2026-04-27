from plots_profiles import make_schwarzschild_profiles, make_kerr_newman_profiles


def main():
    print("Generating Figs. 1 and 2: refractive index profiles...")
    make_schwarzschild_profiles()
    make_kerr_newman_profiles()
    print("Saved profile plots in results/profiles/")


if __name__ == "__main__":
    main()
