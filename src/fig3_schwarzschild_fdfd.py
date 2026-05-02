from plots_fdfd import make_figure_3, make_figure_3_phase


def main():
    print("Generating Fig. 3: Schwarzschild FDFD simulations...")
    make_figure_3()
    print("Saved Schwarzschild FDFD plots in results/fdfd/")
    make_figure_3_phase()
    print("Saved Schwarzschild FDFD phase plots in results/fdfd/")

if __name__ == "__main__":
    main()
