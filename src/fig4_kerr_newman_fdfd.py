from plots_fdfd import make_figure_4, make_figure_4_phase


def main():
    print("Generating Fig. 4: Kerr-Newman FDFD simulations...")
    make_figure_4()
    print("Saved Kerr-Newman FDFD plots in results/fdfd/")
    make_figure_4_phase()
    print("Saved Kerr-Newman FDFD phase plots in results/fdfd/")

if __name__ == "__main__":
    main()
