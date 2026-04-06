"""
2D Finite-Difference Frequency-Domain (FDFD) solver for TM mode (E_z).

Implements the FDFD simulation framework described in the paper (refs. 58,59):
- Helmholtz equation:  nabla^2 E_z + k0^2 * eps_r * E_z = -source
- PML (Perfectly Matched Layer) boundary conditions
- Gaussian beam source

Units: micrometers for length, with wavelength lambda = 0.5 um.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ============================================================================
# PML helper
# ============================================================================

def _pml_sigma(N_pml, d_pml, k0, power=3, R0=1e-8):
    """
    Polynomial PML conductivity profile.
    Returns sigma array of length N_pml.
    """
    if N_pml == 0:
        return np.array([])
    sigma_max = -(power + 1) * np.log(R0) / (2.0 * d_pml)
    d = np.linspace(0, d_pml, N_pml + 1)
    d = 0.5 * (d[:-1] + d[1:])  # cell centers
    return sigma_max * (d / d_pml) ** power


def build_pml_stretch(N, N_pml, dx, k0):
    """
    Build 1D array of complex PML stretch factors s(x) for N cells.
    s = 1 + sigma / (i * k0)  =  1 - i * sigma / k0

    PML regions are at both ends of the domain.
    """
    s = np.ones(N, dtype=complex)
    if N_pml == 0:
        return s
    d_pml = N_pml * dx
    sigma = _pml_sigma(N_pml, d_pml, k0)

    # Left PML: cells 0..N_pml-1, distance increases from center outward
    s[:N_pml] = 1.0 - 1j * sigma[::-1] / k0

    # Right PML: cells N-N_pml..N-1
    s[N - N_pml:] = 1.0 - 1j * sigma / k0

    return s


# ============================================================================
# FDFD solver class
# ============================================================================

class FDFD2D:
    """
    2D FDFD solver for TM mode (E_z polarization).

    Solves:  (d/dx 1/s_x d/dx + d/dy 1/s_y d/dy) E_z + k0^2 eps_r E_z = J_z

    where s_x, s_y are PML stretch factors.
    """

    def __init__(self, Lx, Ly, dx, wavelength, N_pml=10):
        """
        Parameters
        ----------
        Lx, Ly : float
            Physical domain size (excluding PML).
        dx : float
            Grid spacing (same in x and y).
        wavelength : float
            Free-space wavelength.
        N_pml : int
            Number of PML cells on each side.
        """
        self.dx = dx
        self.wavelength = wavelength
        self.k0 = 2.0 * np.pi / wavelength

        # Grid dimensions including PML
        self.Nx_phys = int(round(Lx / dx))
        self.Ny_phys = int(round(Ly / dx))
        self.N_pml = N_pml
        self.Nx = self.Nx_phys + 2 * N_pml
        self.Ny = self.Ny_phys + 2 * N_pml
        self.N = self.Nx * self.Ny

        # Physical coordinates (cell centers, including PML)
        self.x = (np.arange(self.Nx) - N_pml + 0.5) * dx - Lx / 2
        self.y = (np.arange(self.Ny) - N_pml + 0.5) * dx - Ly / 2

        # Default: free space
        self.eps_r = np.ones((self.Nx, self.Ny), dtype=complex)
        self.J_z = np.zeros((self.Nx, self.Ny), dtype=complex)

        # Build PML
        self.sx = build_pml_stretch(self.Nx, N_pml, dx, self.k0)
        self.sy = build_pml_stretch(self.Ny, N_pml, dx, self.k0)

        self.E_z = None  # solution

    def _index2d(self, ix, iy):
        """Flatten 2D index (ix, iy) -> 1D index."""
        return ix * self.Ny + iy

    def set_annular_permittivity(self, center_x, center_y, edges, n_values,
                                  inner_eps=None):
        """
        Set permittivity for concentric annuli.

        Parameters
        ----------
        center_x, center_y : float
            Center of the annular system.
        edges : array
            Annulus boundaries (outer to inner).
        n_values : array
            Refractive index in each annulus.
        inner_eps : complex or None
            Permittivity inside the innermost annulus (absorbing core).
            If None, uses the innermost annulus value with damping.
        """
        XX, YY = np.meshgrid(self.x, self.y, indexing='ij')
        R = np.sqrt((XX - center_x)**2 + (YY - center_y)**2)

        eps = np.ones_like(R, dtype=complex)  # free space outside

        # Fill annuli from outside in
        for i in range(len(n_values)):
            R_outer = edges[i]
            R_inner = edges[i + 1]
            mask = (R <= R_outer) & (R > R_inner)
            eps[mask] = n_values[i] ** 2

        # Inner core (absorbing)
        R_min = edges[-1]
        if inner_eps is None:
            inner_eps = n_values[-1]**2 - 1j * np.pi  # damping factor
        mask_inner = R <= R_min
        eps[mask_inner] = inner_eps

        self.eps_r = eps

    def set_gaussian_beam_source(self, B0, beam_sigma, R0_phys,
                                  center_x=0.0, center_y=0.0):
        """
        Set up a Gaussian beam source propagating in +y direction toward the
        black hole center.

        Paper setup (Fig. 3): beam enters from below, propagates upward.
        Impact parameter B0 is the horizontal (x) offset from the center.
        Absorbing waveguides at x = B0 ± lambda channel the beam from the
        domain edge to the optical black hole edge.

        Parameters
        ----------
        B0 : float
            Impact parameter (x-offset from center), in physical units.
        beam_sigma : float
            Gaussian beam width parameter (delta = lambda/2).
        R0_phys : float
            Outer radius of the optical black hole (physical units).
        center_x, center_y : float
            Center of the black hole.
        """
        lam = self.wavelength
        J = np.zeros((self.Nx, self.Ny), dtype=complex)
        x_beam = center_x + B0

        # Gaussian envelope in x, centered at the beam position
        envelope = np.exp(-((self.x - x_beam)**2) / (2.0 * beam_sigma**2))

        # Truncate to waveguide width (2*lambda total, i.e. ± lambda)
        wg_left = x_beam - lam
        wg_right = x_beam + lam
        wg_mask = (self.x >= wg_left) & (self.x <= wg_right)
        envelope *= wg_mask

        # Place source at bottom edge of domain
        y_start = self.y[self.N_pml + 1]  # just inside the PML
        iy_src = np.argmin(np.abs(self.y - y_start))
        J[:, iy_src] = envelope

        self.J_z = J

        # Apply absorbing waveguide boundaries:
        # absorbing strips at x < wg_left and x > wg_right,
        # but only in the region below the black hole outer edge (y < 0)
        # and outside the optical black hole
        XX, YY = np.meshgrid(self.x, self.y, indexing='ij')
        R = np.sqrt((XX - center_x)**2 + (YY - center_y)**2)

        # Waveguide region: below BH center, outside BH, outside beam channel
        wg_region = (YY < center_y) & (R > R0_phys)
        outside_channel = (XX < wg_left) | (XX > wg_right)
        wg_absorb = wg_region & outside_channel

        self.eps_r[wg_absorb] = 1.0 - 1j * np.pi

    def build_system(self):
        """
        Build the sparse system matrix A such that A @ E_z = b.

        Vectorized construction of the 5-point stencil with PML.
        """
        Nx, Ny, N = self.Nx, self.Ny, self.N
        dx = self.dx
        k0 = self.k0

        # PML stretch at half-grid points (edges between cells)
        sx_fwd = np.ones(Nx, dtype=complex)
        sx_fwd[:-1] = 0.5 * (self.sx[:-1] + self.sx[1:])
        sx_fwd[-1] = self.sx[-1]

        sx_bwd = np.ones(Nx, dtype=complex)
        sx_bwd[1:] = 0.5 * (self.sx[:-1] + self.sx[1:])
        sx_bwd[0] = self.sx[0]

        sy_fwd = np.ones(Ny, dtype=complex)
        sy_fwd[:-1] = 0.5 * (self.sy[:-1] + self.sy[1:])
        sy_fwd[-1] = self.sy[-1]

        sy_bwd = np.ones(Ny, dtype=complex)
        sy_bwd[1:] = 0.5 * (self.sy[:-1] + self.sy[1:])
        sy_bwd[0] = self.sy[0]

        dx2 = dx**2

        # Flatten indices: m = ix*Ny + iy
        ix_all = np.repeat(np.arange(Nx), Ny)  # (N,)
        iy_all = np.tile(np.arange(Ny), Nx)     # (N,)
        m_all = np.arange(N)                     # (N,)

        # Stretched inverse factors for all grid points
        ISX_F = 1.0 / sx_fwd[ix_all]
        ISX_B = 1.0 / sx_bwd[ix_all]
        ISY_F = 1.0 / sy_fwd[iy_all]
        ISY_B = 1.0 / sy_bwd[iy_all]

        # ---- Build COO arrays ----
        rows_list = []
        cols_list = []
        vals_list = []

        # 1) Diagonal
        diag_val = -(ISX_F + ISX_B + ISY_F + ISY_B) / dx2 + k0**2 * self.eps_r.ravel()
        rows_list.append(m_all)
        cols_list.append(m_all)
        vals_list.append(diag_val)

        # 2) x+1 neighbor (m + Ny) -- only for ix < Nx-1
        mask_xp = ix_all < Nx - 1
        idx_src = m_all[mask_xp]
        idx_dst = idx_src + Ny
        rows_list.append(idx_src)
        cols_list.append(idx_dst)
        vals_list.append(ISX_F[mask_xp] / dx2)

        # 3) x-1 neighbor (m - Ny) -- only for ix > 0
        mask_xm = ix_all > 0
        idx_src = m_all[mask_xm]
        idx_dst = idx_src - Ny
        rows_list.append(idx_src)
        cols_list.append(idx_dst)
        vals_list.append(ISX_B[mask_xm] / dx2)

        # 4) y+1 neighbor (m + 1) -- only for iy < Ny-1
        mask_yp = iy_all < Ny - 1
        idx_src = m_all[mask_yp]
        idx_dst = idx_src + 1
        rows_list.append(idx_src)
        cols_list.append(idx_dst)
        vals_list.append(ISY_F[mask_yp] / dx2)

        # 5) y-1 neighbor (m - 1) -- only for iy > 0
        mask_ym = iy_all > 0
        idx_src = m_all[mask_ym]
        idx_dst = idx_src - 1
        rows_list.append(idx_src)
        cols_list.append(idx_dst)
        vals_list.append(ISY_B[mask_ym] / dx2)

        # Assemble
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        vals = np.concatenate(vals_list)

        A = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()
        b = -self.J_z.ravel()

        return A, b

    def solve(self, verbose=True):
        """Build and solve the FDFD system."""
        if verbose:
            print(f"Building FDFD system: {self.Nx}x{self.Ny} = {self.N} unknowns...")

        A, b = self.build_system()

        if verbose:
            print(f"Solving sparse system (nnz={A.nnz})...")

        self.E_z = spsolve(A, b).reshape(self.Nx, self.Ny)

        if verbose:
            print("Done.")

        return self.E_z

    def compute_H_fields(self):
        """Compute H_x, H_y from E_z using finite differences."""
        if self.E_z is None:
            raise ValueError("Solve first.")

        dx = self.dx
        k0 = self.k0

        # H_x = (1/(i*omega*mu)) * dE_z/dy
        # H_y = -(1/(i*omega*mu)) * dE_z/dx
        # With mu=1 and omega = k0: factor = 1/(i*k0)

        H_x = np.zeros_like(self.E_z)
        H_y = np.zeros_like(self.E_z)

        # Central differences
        H_x[:, 1:-1] = (self.E_z[:, 2:] - self.E_z[:, :-2]) / (2 * dx * 1j * k0)
        H_y[1:-1, :] = -(self.E_z[2:, :] - self.E_z[:-2, :]) / (2 * dx * 1j * k0)

        return H_x, H_y

    def compute_poynting(self):
        """Compute time-averaged Poynting vector Re(E x H*/2)."""
        H_x, H_y = self.compute_H_fields()
        E = self.E_z

        # S = Re(E x H*) / 2
        # For TM mode: S_x = Re(E_z * H_y*)/2, S_y = -Re(E_z * H_x*)/2
        S_x = 0.5 * np.real(E * np.conj(H_y))
        S_y = -0.5 * np.real(E * np.conj(H_x))

        return S_x, S_y


# ============================================================================
# High-level simulation functions matching the paper's setup
# ============================================================================

def simulate_schwarzschild(b_inf, P_min=2.0, P0=6.0, n_annuli=16,
                            wavelength=0.5, M=2.5, resolution=15,
                            verbose=True):
    """
    Run FDFD simulation for an optical Schwarzschild black hole.

    Parameters
    ----------
    b_inf : float
        Dimensionless impact parameter.
    P_min : float
        Minimum radius (in units of M).
    P0 : float
        Outer radius (in units of M). Default 6.
    n_annuli : int
        Number of annuli. Default 16 (as in paper).
    wavelength : float
        Wavelength in micrometers. Default 0.5.
    M : float
        Black hole mass scale in micrometers. Default 2.5 (R_S = 5 um).
    resolution : int
        Grid points per wavelength.
    verbose : bool

    Returns
    -------
    fdfd : FDFD2D object with solved fields.
    """
    from ray_tracing import build_schwarzschild_annuli

    # Physical dimensions
    R0 = P0 * M       # outer radius in um
    R_S = 2.0 * M     # Schwarzschild radius in um

    # Domain: 60 lambda x 60 lambda (paper specification)
    L = 60 * wavelength  # 30 um
    dx = wavelength / resolution

    # PML: lambda/5 as in paper
    N_pml = max(int(round(wavelength / (5 * dx))), 4)

    if verbose:
        print(f"Schwarzschild FDFD: b_inf={b_inf}, M={M} um, lambda={wavelength} um")
        print(f"  Domain: {L:.1f} x {L:.1f} um, dx={dx:.4f} um, N_pml={N_pml}")

    fdfd = FDFD2D(L, L, dx, wavelength, N_pml)

    # Build annular system (dimensionless)
    edges_dim, n_values = build_schwarzschild_annuli(b_inf, P_min, P0, n_annuli)
    # Convert edges to physical units
    edges_phys = edges_dim * M

    # Set permittivity
    # Inner core: absorbing (eps = n_inner^2 - i*pi)
    inner_eps = n_values[-1]**2 - 1j * np.pi
    fdfd.set_annular_permittivity(0.0, 0.0, edges_phys, n_values, inner_eps)

    # Set Gaussian beam source (beam enters from below, +y direction)
    B0_phys = b_inf * M  # impact parameter in um (x-offset)
    beam_sigma = wavelength / 2.0  # delta = lambda/2 (paper)
    R0_phys = P0 * M

    fdfd.set_gaussian_beam_source(
        B0_phys, beam_sigma, R0_phys,
        center_x=0.0, center_y=0.0
    )

    # Solve
    fdfd.solve(verbose=verbose)

    return fdfd


def simulate_kerr_newman(a, rho_Q, b_inf, ell_sign, P_min, P0=6.0,
                          n_annuli=21, wavelength=0.5, M=2.5,
                          resolution=15, verbose=True):
    """
    Run FDFD simulation for an optical Kerr-Newman black hole.
    """
    from ray_tracing import build_kn_annuli

    R0 = P0 * M
    L = 60 * wavelength
    dx = wavelength / resolution

    N_pml = max(int(round(wavelength / (5 * dx))), 4)

    if verbose:
        print(f"Kerr-Newman FDFD: a={a}, rho_Q={rho_Q}, b_inf={b_inf}")
        print(f"  Domain: {L:.1f} x {L:.1f} um, dx={dx:.4f} um")

    fdfd = FDFD2D(L, L, dx, wavelength, N_pml)

    edges_dim, n_values = build_kn_annuli(a, rho_Q, b_inf, ell_sign,
                                           P_min, P0, n_annuli)
    edges_phys = edges_dim * M

    inner_eps = n_values[-1]**2 - 1j * np.pi
    fdfd.set_annular_permittivity(0.0, 0.0, edges_phys, n_values, inner_eps)

    B0_phys = b_inf * M
    beam_sigma = wavelength / 2.0
    R0_phys = P0 * M

    fdfd.set_gaussian_beam_source(
        B0_phys, beam_sigma, R0_phys
    )

    fdfd.solve(verbose=verbose)
    return fdfd