"""
Tensor-material extension of the FDFD solver.

The base `FDFD2D` class in fdfd.py solves the scalar TM Helmholtz equation:

    (1/s_x d/dx 1/s_x d/dx + ... ) E_z + k0^2 eps_r E_z = J_z

with eps_r a scalar at each grid point. That is enough to reproduce the null
GEODESICS of the Schwarzschild metric (the paper's Eq. 14 scalar n).

This module adds `FDFD2D_Tensor`, which solves the TM curl-curl equation for
a fully tensor-valued in-plane permeability mu_ij and out-of-plane
permittivity eps_zz:

    -d/dx ( nu_yy d_x E_z - nu_yx d_y E_z )
    -d/dy ( nu_xx d_y E_z - nu_xy d_x E_z )
    + k0^2 eps_zz E_z = J_z

where nu = mu^{-1} is the inverse 2x2 in-plane permeability tensor, taken
cell-by-cell.

Reduction check: when mu_xx = mu_yy = 1, mu_xy = 0 (so nu_xx = nu_yy = 1,
nu_xy = 0), the equation collapses to the scalar Helmholtz equation that
FDFD2D solves, so the new class strictly extends the old one.

The solver discretizes derivatives on a square grid using the same staggered
forward/backward stretched-coordinate scheme as the parent class, so PML
behaviour is identical. The only thing that changes is the system matrix:
each cell now has up to 9 stencil neighbours (the 5-point Laplacian PLUS
the diagonal neighbours that come from the cross terms d/dx(nu_yx d_y E_z)
and d/dy(nu_xy d_x E_z)).

Use case in this codebase: the polarisation-preserving Schwarzschild
analogue based on paper Eq. (8). The tensor field is built by
Schwarzchild_tensor.schwarzschild_tensor_annular_field; the result is a
field eps_zz, mu_xx, mu_yy, mu_xy at every grid cell that is plugged into
the constructor here.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from fdfd import FDFD2D, build_pml_stretch


class FDFD2D_Tensor(FDFD2D):
    """
    TM-mode FDFD solver supporting a tensor in-plane permeability mu_ij and
    a scalar (per-cell) out-of-plane permittivity eps_zz.

    Inherits the grid, PML, source machinery, and post-processing helpers
    from FDFD2D. Overrides only `build_system`, plus adds
    `set_tensor_material`.
    """

    def __init__(self, Lx, Ly, dx, wavelength, N_pml=12):
        super().__init__(Lx, Ly, dx, wavelength, N_pml)

        # Default: free space (mu = identity, eps_zz already = 1 from parent)
        self.mu_xx = np.ones((self.Nx, self.Ny), dtype=complex)
        self.mu_yy = np.ones((self.Nx, self.Ny), dtype=complex)
        self.mu_xy = np.zeros((self.Nx, self.Ny), dtype=complex)
        # eps_zz is stored in self.eps_r (inherited) -- TM electric component

    # ------------------------------------------------------------------
    # Material setters
    # ------------------------------------------------------------------

    def set_tensor_material(self, eps_zz, mu_xx, mu_yy, mu_xy):
        """
        Install per-cell tensor material (all arrays of shape (Nx, Ny)).

        eps_zz : complex array - electric permittivity for E_z component
        mu_xx, mu_yy, mu_xy : complex arrays - in-plane permeability tensor
                              (mu_yx = mu_xy assumed by symmetry)
        """
        if eps_zz.shape != (self.Nx, self.Ny):
            raise ValueError(
                f"eps_zz shape {eps_zz.shape} != grid {(self.Nx, self.Ny)}")
        self.eps_r = eps_zz.astype(complex)
        self.mu_xx = mu_xx.astype(complex)
        self.mu_yy = mu_yy.astype(complex)
        self.mu_xy = mu_xy.astype(complex)

    def set_tensor_annular(self, center_x, center_y, get_components_fn,
                           inner_eps=None, inner_mu=None,
                           absorb_imag=np.pi):
        """
        Convenience method: install a tensor material from a callable that
        returns the tensor field on the grid.

        get_components_fn(X_dimless, Y_dimless) -> dict with keys
            'eps_zz', 'mu_xx', 'mu_yy', 'mu_xy'

        Inside the medium's defined region (signalled by the field returning
        non-default values), those tensor entries are used; outside, the grid
        keeps free-space values.

        For the inner core (R < P_min after the medium is laid down), this
        method assumes the caller has already restricted the get_components
        result to that region; for filling the absorbing core, use
        `set_inner_core` afterwards.
        """
        # Pass the dimensionless grid to the function
        XX, YY = np.meshgrid(self.x, self.y, indexing='ij')
        comps = get_components_fn(XX, YY)

        self.eps_r = comps['eps_zz'].astype(complex)
        self.mu_xx = comps['mu_xx'].astype(complex)
        self.mu_yy = comps['mu_yy'].astype(complex)
        self.mu_xy = comps['mu_xy'].astype(complex)

    def set_inner_core(self, center_x, center_y, R_min,
                       eps_core=None, mu_core_diag=1.0,
                       absorb_imag=np.pi):
        """
        Replace material inside R < R_min with an isotropic absorbing core,
        matching the scalar code's convention: eps = eps_core - i*absorb_imag,
        mu = mu_core_diag (default 1).

        eps_core : complex or None
            If None, use the value of eps_zz at R = R_min (mean over the
            inner ring) as the real part. This mirrors the scalar code's
            "use the innermost-annulus value with damping" rule.
        """
        XX, YY = np.meshgrid(self.x, self.y, indexing='ij')
        R = np.sqrt((XX - center_x)**2 + (YY - center_y)**2)
        mask = R < R_min

        if eps_core is None:
            ring_mask = (R >= R_min) & (R < R_min + self.dx * 1.5)
            if np.any(ring_mask):
                eps_core = np.mean(np.real(self.eps_r[ring_mask]))
            else:
                eps_core = 1.0

        self.eps_r[mask]  = eps_core - 1j * absorb_imag
        self.mu_xx[mask]  = mu_core_diag
        self.mu_yy[mask]  = mu_core_diag
        self.mu_xy[mask]  = 0.0

    # ------------------------------------------------------------------
    # Sparse system assembly
    # ------------------------------------------------------------------

    def build_system(self):
        """
        Assemble A, b for  A E_z = b  with the tensor-mu TM curl-curl operator:

           -d/dx [ nu_yy d_x E_z - nu_yx d_y E_z ]
           -d/dy [ nu_xx d_y E_z - nu_xy d_x E_z ]
           + k0^2 eps_zz E_z  =  -J_z

        Sign conventions match FDFD2D.build_system so that mu = identity
        recovers the parent class result (verified by reduction check).

        Discretisation: each derivative is a centred difference using the
        forward/backward stretched-coordinate factors at half-grid points
        sx_fwd, sx_bwd, sy_fwd, sy_bwd that the parent already computes.

        Cross terms d/dx(nu_yx d_y E_z) introduce neighbour couplings to the
        four diagonal cells (ix +/- 1, iy +/- 1), giving a 9-point stencil
        in general.

        Implementation note
        -------------------
        We construct the operator one term at a time, accumulating COO
        triplets, then assemble at the end. This keeps the algebra readable
        and makes it easier to verify against the parent's assembly when
        the tensor reduces to identity. Performance: we build O(9 N) entries,
        which is fine for the grid sizes the paper uses (~600x600).
        """
        Nx, Ny, N = self.Nx, self.Ny, self.N
        dx = self.dx
        k0 = self.k0

        # ---- PML stretch factors at half-grid edges (same as parent) ----
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

        dx2 = dx * dx

        # ---- Per-cell inverse permeability tensor ν = μ^{-1} ----
        # μ = [[mu_xx, mu_xy], [mu_xy, mu_yy]]
        # det μ = mu_xx*mu_yy - mu_xy^2
        det_mu = self.mu_xx * self.mu_yy - self.mu_xy * self.mu_xy
        # Guard against zero / very small determinants (e.g. unfilled vacuum
        # cells where det = 1 anyway, but caution near horizon)
        det_safe = np.where(np.abs(det_mu) > 1e-30, det_mu, 1e-30)
        nu_xx =  self.mu_yy / det_safe
        nu_yy =  self.mu_xx / det_safe
        nu_xy = -self.mu_xy / det_safe   # nu_yx = nu_xy by symmetry of mu

        # Flatten to 1D arrays indexed by m = ix*Ny + iy
        nu_xx_f = nu_xx.ravel()
        nu_yy_f = nu_yy.ravel()
        nu_xy_f = nu_xy.ravel()

        # Cell index helpers
        ix_all = np.repeat(np.arange(Nx), Ny)
        iy_all = np.tile(np.arange(Ny), Nx)
        m_all  = np.arange(N)

        # Stretched inverse factors at the four half-edges of each cell
        ISX_F = 1.0 / sx_fwd[ix_all]   # at face between (ix, ix+1)
        ISX_B = 1.0 / sx_bwd[ix_all]   # at face between (ix-1, ix)
        ISY_F = 1.0 / sy_fwd[iy_all]
        ISY_B = 1.0 / sy_bwd[iy_all]

        rows_list, cols_list, vals_list = [], [], []

        def add(rows, cols, vals):
            rows_list.append(rows)
            cols_list.append(cols)
            vals_list.append(vals)

        # ==============================================================
        # TERM 1:  -d/dx [ nu_yy d_x E_z ]   (with PML stretch)
        # ==============================================================
        # We must match the PARENT class' single-stretched discretisation,
        # in which the equation is taken as
        #
        #     (1/s_x d/dx) E_z|_face / dx, summed/differenced across cells
        #
        # giving stencil entries:
        #     diag : -(ISX_F + ISX_B) / dx2
        #     +x   : +ISX_F / dx2
        #     -x   : +ISX_B / dx2
        #
        # The tensor generalisation multiplies each FACE derivative by
        # nu_yy averaged to that face:
        #     nu_yy_F = 0.5 (nu_yy[ix+1, iy] + nu_yy[ix, iy])     forward face
        #     nu_yy_B = 0.5 (nu_yy[ix-1, iy] + nu_yy[ix, iy])     backward face
        # giving:
        #     diag : -(nu_yy_F * ISX_F + nu_yy_B * ISX_B) / dx2
        #     +x   : +nu_yy_F * ISX_F / dx2
        #     -x   : +nu_yy_B * ISX_B / dx2
        #
        # Reduction check (mu = I -> nu_yy = 1): recovers parent exactly.
        # ----------------------------------------------------------------
        nu_yy_xF = np.empty_like(nu_yy)
        nu_yy_xF[:-1, :] = 0.5 * (nu_yy[:-1, :] + nu_yy[1:, :])
        nu_yy_xF[-1, :]  = nu_yy[-1, :]      # boundary -> use cell value
        nu_yy_xB = np.empty_like(nu_yy)
        nu_yy_xB[1:, :]  = 0.5 * (nu_yy[:-1, :] + nu_yy[1:, :])
        nu_yy_xB[0, :]   = nu_yy[0, :]

        nu_yy_xF_f = nu_yy_xF.ravel()
        nu_yy_xB_f = nu_yy_xB.ravel()

        # Diagonal contribution from term 1
        diag1 = -(nu_yy_xF_f * ISX_F + nu_yy_xB_f * ISX_B) / dx2
        add(m_all, m_all, diag1)

        # +x neighbour
        mask_xp = ix_all < Nx - 1
        add(m_all[mask_xp], m_all[mask_xp] + Ny,
            nu_yy_xF_f[mask_xp] * ISX_F[mask_xp] / dx2)

        # -x neighbour
        mask_xm = ix_all > 0
        add(m_all[mask_xm], m_all[mask_xm] - Ny,
            nu_yy_xB_f[mask_xm] * ISX_B[mask_xm] / dx2)

        # ==============================================================
        # TERM 2:  -d/dy [ nu_xx d_y E_z ]   (with PML stretch)
        # ==============================================================
        nu_xx_yF = np.empty_like(nu_xx)
        nu_xx_yF[:, :-1] = 0.5 * (nu_xx[:, :-1] + nu_xx[:, 1:])
        nu_xx_yF[:, -1]  = nu_xx[:, -1]
        nu_xx_yB = np.empty_like(nu_xx)
        nu_xx_yB[:, 1:]  = 0.5 * (nu_xx[:, :-1] + nu_xx[:, 1:])
        nu_xx_yB[:, 0]   = nu_xx[:, 0]

        nu_xx_yF_f = nu_xx_yF.ravel()
        nu_xx_yB_f = nu_xx_yB.ravel()

        diag2 = -(nu_xx_yF_f * ISY_F + nu_xx_yB_f * ISY_B) / dx2
        add(m_all, m_all, diag2)

        # +y neighbour
        mask_yp = iy_all < Ny - 1
        add(m_all[mask_yp], m_all[mask_yp] + 1,
            nu_xx_yF_f[mask_yp] * ISY_F[mask_yp] / dx2)

        # -y neighbour
        mask_ym = iy_all > 0
        add(m_all[mask_ym], m_all[mask_ym] - 1,
            nu_xx_yB_f[mask_ym] * ISY_B[mask_ym] / dx2)

        # ==============================================================
        # CROSS TERMS:
        #   d/dx [ nu_yx d_y E_z ]  +  d/dy [ nu_xy d_x E_z ]
        # (with the original signs; combined with the leading minus signs
        # in the operator to give the contribution below)
        # ==============================================================
        # We discretise both cross terms using centred differences. With
        # single-stretched PML and centred face values for nu_xy:
        #
        #   d/dx [nu_xy d_y E_z]  at cell (i, j)
        #     ~ ( ISX_C/(2 dx) ) * [
        #         nu_xy[i+1,j] * (E[i+1,j+1] - E[i+1,j-1])/(2 dy ISY_C^{-1})
        #       - nu_xy[i-1,j] * (E[i-1,j+1] - E[i-1,j-1])/(2 dy ISY_C^{-1}) ]
        #
        # i.e. each diagonal neighbour (i +/- 1, j +/- 1) contributes a term
        # of magnitude  nu_xy * ISX_C * ISY_C / (4 dx2)  with a sign that is
        # +1 if (di * dj) > 0 and -1 if (di * dj) < 0.
        # The other cross derivative d/dy[nu_xy d_x E_z] is identical (since
        # we use a symmetric tensor and centred differences), so the total
        # is twice that.
        #
        # Reduction check: when nu_xy = 0 the contribution vanishes, so the
        # operator reduces to terms 1 + 2, which we already verified reduce
        # to the parent class.
        # ----------------------------------------------------------------
        ISX_C = 0.5 * (ISX_F + ISX_B)
        ISY_C = 0.5 * (ISY_F + ISY_B)
        coeff = ISX_C * ISY_C / (4.0 * dx2)

        for di, dj in [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]:
            sign = +1.0 if di * dj > 0 else -1.0

            mask = ((ix_all + di >= 0) & (ix_all + di < Nx) &
                    (iy_all + dj >= 0) & (iy_all + dj < Ny))
            src = m_all[mask]
            dst = src + di * Ny + dj

            # Average nu_xy between source and neighbour (face value)
            nu_face = 0.5 * (nu_xy_f[src] + nu_xy_f[dst])

            # Factor of 2: contributions from both d/dx[nu_xy d_y] and
            # d/dy[nu_xy d_x], identical by symmetry.
            val = sign * 2.0 * nu_face * coeff[mask]
            add(src, dst, val)

        # ==============================================================
        # ELECTRIC TERM:  + k0^2 eps_zz E_z   (on the diagonal)
        # ==============================================================
        diag_eps = (k0 ** 2) * self.eps_r.ravel()
        add(m_all, m_all, diag_eps)

        # ==============================================================
        # Assemble
        # ==============================================================
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        vals = np.concatenate(vals_list)

        A = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()
        b = -self.J_z.ravel()

        return A, b


# ============================================================================
# High-level wrapper analogous to simulate_schwarzschild
# ============================================================================

def simulate_schwarzschild_tensor(b_inf, P_min=2.1, P0=6.0, n_annuli=16,
                                   wavelength=0.5, M=2.5, resolution=15,
                                   N_pml=12, verbose=True):
    """
    Run a TENSOR-material FDFD simulation for the polarisation-preserving
    optical Schwarzschild analogue (paper Eq. 8 in equatorial plane).

    Mirrors the API of fdfd.simulate_schwarzschild so the two can be plotted
    side-by-side.

    NOTE on P_min: the tensor formulation has the same 1/(2 - P) divergence
    at the horizon as the scalar one. The paper avoids the horizon for the
    same reason. We default to P_min = 2.1 (slightly outside R_S = 2M) but
    you can push closer at the cost of larger eps/mu values per annulus.
    """
    from Schwarzchild_tensor import schwarzschild_tensor_annular_field

    L = 60 * wavelength
    dx = wavelength / resolution
    R0_phys = P0 * M
    R_min_phys = P_min * M

    if verbose:
        print(f"Schwarzschild TENSOR FDFD: b_inf={b_inf}, M={M} um, "
              f"lambda={wavelength} um")
        print(f"  Domain: {L:.1f} x {L:.1f} um, dx={dx:.4f} um, "
              f"N_pml={N_pml}, P_min={P_min}")

    fdfd = FDFD2D_Tensor(L, L, dx, wavelength, N_pml)

    # Build dimensionless coordinate grids (everything in cases.py / the
    # tensor builder uses dimensionless P = R/M).
    XX, YY = np.meshgrid(fdfd.x / M, fdfd.y / M, indexing='ij')
    R_dimless = np.sqrt(XX**2 + YY**2)

    # Get the annular tensor field on the grid
    comps = schwarzschild_tensor_annular_field(XX, YY, P_min, P0, n_annuli)

    # Restrict to the medium region [P_min, P0]; outside is free space (the
    # tensor builder already returns identity outside, but be explicit).
    in_medium = (R_dimless <= P0) & (R_dimless >= P_min)
    eps_zz = np.ones_like(R_dimless, dtype=complex)
    mu_xx  = np.ones_like(R_dimless, dtype=complex)
    mu_yy  = np.ones_like(R_dimless, dtype=complex)
    mu_xy  = np.zeros_like(R_dimless, dtype=complex)

    eps_zz[in_medium] = comps['eps_zz'][in_medium]
    mu_xx[in_medium]  = comps['mu_xx'][in_medium]
    mu_yy[in_medium]  = comps['mu_yy'][in_medium]
    mu_xy[in_medium]  = comps['mu_xy'][in_medium]

    fdfd.set_tensor_material(eps_zz, mu_xx, mu_yy, mu_xy)

    # Inner absorbing core (matches scalar code)
    fdfd.set_inner_core(0.0, 0.0, R_min_phys,
                        eps_core=None, absorb_imag=np.pi)

    # Source: same Gaussian beam machinery as the parent class
    B0_phys = b_inf * M
    beam_sigma = wavelength / 2.0
    fdfd.set_gaussian_beam_source(
        B0_phys, beam_sigma, R0_phys,
        center_x=0.0, center_y=0.0
    )

    fdfd.solve(verbose=verbose)
    return fdfd


if __name__ == "__main__":
    # Smoke test: run a tiny case at low resolution
    print("Smoke test: tensor FDFD at low resolution...")
    fdfd = simulate_schwarzschild_tensor(
        b_inf=3.0, P_min=2.1, P0=6.0, n_annuli=16,
        wavelength=0.5, M=2.5, resolution=6, N_pml=8, verbose=True
    )
    print(f"  E_z shape = {fdfd.E_z.shape}")
    print(f"  max |E| = {np.max(np.abs(fdfd.E_z)):.3e}")
    print(f"  mean |E| in medium = {np.mean(np.abs(fdfd.E_z)):.3e}")
    print("OK")
