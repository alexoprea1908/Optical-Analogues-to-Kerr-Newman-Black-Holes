"""
2D Finite-Difference Frequency-Domain (FDFD) solver for TM mode (E_z).

Implements the FDFD simulation framework described in the paper (refs. 58,59):
- Helmholtz equation:  nabla^2 E_z + k0^2 * eps_r * E_z = -source
- Stretched-coordinate PML (Shin & Fan 2012)
- Gaussian beam source channelled by thin absorbing waveguides

Units: micrometers for length, with wavelength lambda = 0.5 um.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ============================================================================
# PML helper, this is the perfectly matching layer placed at the boundaries.
# ============================================================================

def _pml_sigma(N_pml, d_pml, k0, power=3, R0=1e-8):#why 1e-8? The parameter R0 represents the desired reflection coefficient at the outer edge of the PML layer. A value of 1e-8 means that we want the PML to reflect only a very small fraction (10^-8) of the incident wave back into the computational domain, which ensures that the PML effectively absorbs outgoing waves with minimal reflection.
    """
    Polynomial PML conductivity profile.
    Returns sigma array of length N_pml.
    """
    if N_pml == 0:
        return np.array([])
    sigma_max = -(power + 1) * np.log(R0) / (2.0 * d_pml)# The maximum conductivity sigma_max is calculated based on the desired reflection coefficient R0 and the thickness of the PML layer d_pml. The formula ensures that the PML effectively absorbs outgoing waves with minimal reflection back into the computational domain.
    d = np.linspace(0, d_pml, N_pml + 1)
    d = 0.5 * (d[:-1] + d[1:])  # cell centers
    #example: if d_pml = 1 and N_pml = 4, then d will be [0.125, 0.375, 0.625, 0.875], which are the centers of the four PML cells that span from 0 to 1.
    return sigma_max * (d / d_pml) ** power# The conductivity profile sigma is calculated as a polynomial function of the distance d from the inner edge of the PML layer. The power parameter controls how quickly the conductivity increases within the PML, with higher powers leading to a more gradual increase. This profile helps to ensure that waves are absorbed effectively as they propagate through the PML region, minimizing reflections back into the main computational domain.


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

    return s #s is an array of complex stretch factors that modify the spatial derivatives in the FDFD equations to implement the PML. The stretch factors are designed to absorb outgoing waves effectively, with the conductivity profile sigma controlling how the absorption increases within the PML regions at both ends of the domain.


# ============================================================================
# FDFD solver class
# ============================================================================

class FDFD2D:
    """
    2D FDFD solver for TM mode (E_z polarization).

    Solves:  (d/dx 1/s_x d/dx + d/dy 1/s_y d/dy) E_z + k0^2 eps_r E_z = J_z

    where s_x, s_y are PML stretch factors.
    """

    def __init__(self, Lx, Ly, dx, wavelength, N_pml=12):
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
            Number of PML cells on each side. Defaults to 12, which gives
            adequate absorption for the Shin/Fan SC-PML formulation. The
            paper's stated "lambda/5" is unrealistically thin numerically;
            we trade a slightly larger domain for clean boundaries.
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
        self.x = (np.arange(self.Nx) - N_pml + 0.5) * dx - Lx / 2#why is it an array?
        self.y = (np.arange(self.Ny) - N_pml + 0.5) * dx - Ly / 2
#example: if Lx = 10, dx = 1, and N_pml = 2, then self.Nx_phys = 10, self.Nx = 14, and self.x will be an array of length 14 that represents the x-coordinates of the cell centers, including the PML regions. The coordinates will range from -5 to +5 (the physical domain) with additional points for the PML on either side.
#does this encode pml centers as well? yes, the coordinates in self.x and self.y include the centers of the grid cells for both the physical domain and the PML regions. The PML regions are accounted for by the offset of N_pml in the calculation of the coordinates, ensuring that the grid points correctly represent the entire computational domain, including the PML layers at both ends.
        # Default: free space, we initialize e_r as an array of 1, and J_z as 0s
        self.eps_r = np.ones((self.Nx, self.Ny), dtype=complex)
        self.J_z = np.zeros((self.Nx, self.Ny), dtype=complex)

        # Build PML, we get the stretch factors on x and y, using the previous helper function. These stretch factors will be used to modify the spatial derivatives in the FDFD equations to implement the PML.
        self.sx = build_pml_stretch(self.Nx, N_pml, dx, self.k0)
        self.sy = build_pml_stretch(self.Ny, N_pml, dx, self.k0)
        #stretch factors are just for absorbing boundaries, in order to simulate the inner region of the black hole, and the open space outside;
        self.E_z = None  # solution

    def _index2d(self, ix, iy):
        """Flatten 2D index (ix, iy) -> 1D index."""
        return ix * self.Ny + iy
    #example: if self.Ny = 5, then the 2D index (ix=2, iy=3) would be flattened to the 1D index 2*5 + 3 = 13. This indexing scheme allows us to represent the 2D grid of E_z values as a 1D array when we construct the sparse matrix for the FDFD equations.
#give me a concrete example of flattening: if we have a 2D grid with dimensions Nx = 3 and Ny = 4, the 2D indices would be:
#(0, 0), (0, 1), (0, 2), (0, 3)
#(1, 0), (1, 1), (1, 2), (1, 3)
#(2, 0), (2, 1), (2, 2), (2, 3)
#Using the _index2d function, we can flatten these 2D indices to 1D indices as follows:
#(0, 0) -> 0*4 + 0 = 0
#(0, 1) -> 0*4 + 1 = 1
#(0, 2) -> 0*4 + 2 = 2
#(0, 3) -> 0*4 + 3 = 3
#(1, 0) -> 1*4 + 0 = 4
#(1, 1) -> 1*4 + 1 = 5
#(1, 2) -> 1*4 + 2 = 6  
#very smart method, put reference of flattening method in report.
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
        #what are self.x and self.y? they are the physical coordinates of the grid points in the x and y directions, respectively. They are 1D arrays that represent the positions of the grid points along each axis, including the PML regions. These coordinates are used to calculate the radius R from the center for each grid point, which is essential for determining which annulus each grid point belongs to and assigning the appropriate permittivity values based on the refractive index profile defined by n_values and edges.
        #can sel.x and self.y have different lengths? no, they must have lengths self.Nx and self.Ny respectively, which correspond to the total number of grid points in the x and y directions, including the PML regions. This ensures that when we create the meshgrid XX, YY, we get 2D arrays that correctly represent the coordinates of each grid point in the 2D domain.
        XX, YY = np.meshgrid(self.x, self.y, indexing='ij')# Create 2D arrays of x and y coordinates for each grid point. The indexing='ij' argument ensures that the first dimension corresponds to x and the second to y, which is important for correctly calculating the radius R from the center.
        #example: if self.x = [x0, x1, x2] and self.y = [y0, y1], then XX will be [[x0, x0], [x1, x1], [x2, x2]] and YY will be [[y0, y1], [y0, y1], [y0, y1]]. This allows us to compute the radius R for each grid point relative to the center of the annular system.
        #so the 2D grid looks like this:
        #(x0, y0) (x0, y1) 
        #(x1, y0) (x1, y1)
        #(x2, y0) (x2, y1) . nice ok. XX takes the x coordinates and YY takes the y coordinates, so when we evaluate XX and YY simultaneously, we get the coordinates of each grid point in the 2D domain.
        #what does XX[i, j] represent? it represents the x-coordinate of the grid point at position (i, j) in the 2D grid. Similarly, YY[i, j] represents the y-coordinate of the same grid point. This way, we can calculate the radius R from the center for each grid point using these coordinate arrays.
        R = np.sqrt((XX - center_x)**2 + (YY - center_y)**2)#R is an array that contains the distance from the center of the annular system to each grid point in the 2D domain. This radius is used to determine which annulus each grid point belongs to based on the edges of the annuli, and to assign the appropriate refractive index values accordingly.
        #R is a 2D array where each element R[i, j] gives the distance from the center (center_x, center_y) to the grid point at coordinates (XX[i, j], YY[i, j]). This allows us to classify each grid point into the correct annulus based on its distance from the center and the defined edges of the annuli.
        eps = np.ones_like(R, dtype=complex)  # free space outside
        #example: if R is a 2D array of shape (Nx, Ny), then eps will also be a 2D array of the same shape, initialized to 1 (representing free space) at all grid points. We will then modify the values in eps based on the annular structure defined by the edges and n_values, assigning different permittivity values to different regions of the grid according to their distance from the center.

        # Fill annuli from outside in
        for i in range(len(n_values)):#n_values is an array containing the refraction index at each distance from the center. it's length corresponds to the number of annuli, and each value in n_values corresponds to the refractive index for the annulus defined by the edges. The loop iterates through each annulus, using the edges to determine which grid points belong to that annulus and assigning the corresponding permittivity value based on n_values.
            R_outer = edges[i]
            R_inner = edges[i + 1]
            mask = (R <= R_outer) & (R > R_inner)#1 if at distance R from the center we are in the annuli defined by R_outer and R_inner, 0 otherwise. This mask is used to identify which grid points belong to the current annulus being processed in the loop.
            eps[mask] = n_values[i] ** 2#for each annulus, eps=n^2 in that annulus.

        # Inner core (absorbing)
        R_min = edges[-1]
        if inner_eps is None:
            inner_eps = n_values[-1]**2 - 1j * np.pi  # damping factor
        mask_inner = R <= R_min
        eps[mask_inner] = inner_eps

        self.eps_r = eps#self.eps_r is the 2D array that represents the relative permittivity at each grid point in the computational domain. After running this function, self.eps_r will contain the permittivity values corresponding to the annular structure defined by the edges and n_values, as well as the inner core if specified. This array will be used in the FDFD equations to solve for the electric field distribution E_z based on the defined refractive index profile of the optical black hole analog.

    def set_gaussian_beam_source(self, B0, beam_sigma, R0_phys,
                                  center_x=0.0, center_y=0.0,
                                  wg_strip_width=None):
        """
        Set up a Gaussian beam source propagating in +y direction toward the
        black hole center.

        Paper setup (Fig. 3): beam enters from below, propagates upward.
        Impact parameter B0 is the horizontal (x) offset from the center.

        Per the paper's Methods section, two thin absorbing waveguide STRIPS
        are placed immediately to the left and right of the beam channel
        (x in [B0 - lambda, B0 + lambda]), running from the bottom of the
        physical domain up to the outer edge of the optical black hole.
        These prevent the Gaussian from spreading sideways before reaching
        the BH but, crucially, do NOT absorb the rest of the lower half of
        the domain (where backscatter and side-scatter must remain visible).

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
        wg_strip_width : float or None
            Thickness of each absorbing strip. Defaults to 2*lambda, which
            is enough to suppress lateral leakage from the beam but leaves
            most of the domain transparent.
        """
        lam = self.wavelength# this is lambda, the wavelength of light in free space. The beam width parameter beam_sigma is typically set to lambda/2, which defines the width of the Gaussian beam. 
        if wg_strip_width is None:
            wg_strip_width = 2.0 * lam#set the wifth of absorbing stripd to 2lambda

        J = np.zeros((self.Nx, self.Ny), dtype=complex)#2D array representing the source term J_z in the FDFD equations. We will populate this array with the Gaussian beam profile at the appropriate location in the grid. 
        x_beam = center_x + B0

        # Gaussian envelope in x, centered at the beam position
        envelope = np.exp(-((self.x - x_beam)**2) / (2.0 * beam_sigma**2))#it is a 1D array that 
        #example: envelope[i] = exp(-((self.x[i] - x_beam)**2) / (2.0 * beam_sigma**2)). Same length as self.x. Each value represents the value of the envelope at the corresponding x coordinate(from self.x), which is the center of each cell.

        # Truncate to waveguide channel (full width 2*lambda, i.e. ± lambda)
        wg_left = x_beam - lam#left end coordinate of absorbing strip
        wg_right = x_beam + lam#right end coordinate of absorbing strip
        wg_mask = (self.x >= wg_left) & (self.x <= wg_right)# mask which is 1 if the x coordinate is within the waveguide, 0 if not
        envelope *= wg_mask#envelope now becomes 0 outside the waveguide channel, and retains its Gaussian shape within the channel.

        # Place source at bottom edge of physical domain (just inside PML)
        y_start = self.y[self.N_pml + 1]#starts at bottom, from the first cell outside the absorber.
        iy_src = np.argmin(np.abs(self.y - y_start))#this find the index of self.y which gives y_start, such that self.y[iy_src]=y_start.
        J[:, iy_src] = envelope#we have a horizontal line og gaussian source at the bottom of the domain, right above the pml at Y_start.
        self.J_z = J
        #note that J is a 2D array, which is now all 0 except for the row corresponding to iy_src, which contains the Gaussian beam profile defined by the envelope. 
        
        # ---- Thin absorbing waveguide STRIPS bordering the beam channel ----
        #
        # Two narrow vertical absorbers, one just left of x = wg_left and one
        # just right of x = wg_right. They run from the bottom of the physical
        # domain (y = -Ly/2) up to where the beam enters the BH outer circle.
        XX, YY = np.meshgrid(self.x, self.y, indexing='ij')
        #again for computational speed;
        R = np.sqrt((XX - center_x)**2 + (YY - center_y)**2)#R[i][j]=distance from center to grid point (i,j).

        # Vertical extent: from bottom up to the BH entry point.
        # The BH outer edge intersects x = x_beam at
        #     y_entry = center_y - sqrt(R0_phys^2 - B0^2)   (when |B0| < R0)
        # If |B0| >= R0 the beam never enters the BH; cap the strips at y=0.
        if abs(B0) < R0_phys:#if the beam enters the BH;
            y_entry = center_y - np.sqrt(R0_phys**2 - B0**2)#easy pythagoras, this is the y coordinate at which the beam enters the BH starting with an initail impact parameter, B0.
        else:
            y_entry = center_y#if the beam never enters the BH, we still want to place the absorbing strips up to the center line (y=0) to prevent lateral spreading of the beam before it reaches the BH region.

        # Strip 1: left side, x in (wg_left - w, wg_left)
        # Strip 2: right side, x in (wg_right, wg_right + w)
        #wg_left=left x coordinate of the strip.
        #wg_right=right x coordinate of the strip.
        left_strip = (XX > wg_left - wg_strip_width) & (XX < wg_left)#2D array. left_strip[i][j] is 1 if the grid point (i,j) is in the left absorbing strip region, which is defined as the area between x = wg_left - wg_strip_width and x = wg_left.
        #this is 1 for grid points which are in the left strip region, 0 otherwise.
        right_strip = (XX > wg_right) & (XX < wg_right + wg_strip_width)#same procedure
        vertical_extent = (YY < y_entry) & (YY > -self.Ny_phys * self.dx / 2)
        #2D array, vertical_extent[i][j] is 1 if the grid point (i,j) is below the y_entry line and above the bottom edge of the physical domain, which defines the vertical extent of the absorbing strips. 
        # Make absolutely sure we never overwrite the black hole interior
        outside_BH = R > R0_phys

        wg_absorb = vertical_extent & outside_BH & (left_strip | right_strip)#2d array, wg_absorb[i][j] is 1 if the grid point (i,j) is within the vertical extent of the absorbing strips, outside the black hole region, and within either the left or right strip region. This mask identifies the grid points where we want to set the permittivity to create the absorbing waveguide strips.
        self.eps_r[wg_absorb] = 1.0 - 1j * np.pi#we set the permittivity in the absorbing strip regions to a complex value with a negative imaginary part, which creates an absorbing medium that suppresses the lateral spreading of the Gaussian beam as it propagates towards the black hole. 

    def build_system(self):
        """
        Build the sparse system matrix A such that A @ E_z = b.

        Vectorized construction of the 5-point stencil with PML.
        """
        #we are solving the Hekmholtz equation in the form A*Ez=B. A=(d/dx 1/s_x d/dx + d/dy 1/s_y d/dy) + k0^2 eps_r, and B=-J_z.
        #notice that J is a 2D array, but we will flatten it to a 1D array when we construct b. Similarly, E_z will be solved as a 1D array and then reshaped back to 2D.
        #why does this flattening preserve equation
        #concrete example of A.
        Nx, Ny, N = self.Nx, self.Ny, self.N
        dx = self.dx
        k0 = self.k0

        # PML stretch at half-grid points (edges between cells)
        sx_fwd = np.ones(Nx, dtype=complex)#1D array currently containg Nx ones. sx_fwd[i] represents the stretch factor at the forward edge of the cell in the x direction;
        sx_fwd[:-1] = 0.5 * (self.sx[:-1] + self.sx[1:])#we average the stretch factors at the cell centers to get the stretch factor at the edge between cells. sx_fwd[i] = 0.5 * (sx[i] + sx[i+1]) for i=0 to Nx-2. This gives us the stretch factor at the forward edge of each cell
        sx_fwd[-1] = self.sx[-1]#the last edge (at the right boundary) uses the stretch factor of the last cell center, since there is no cell beyond it.

        #we now do the same thing backwards to get the stretch factor at the backward edge of each cell. 
        sx_bwd = np.ones(Nx, dtype=complex)
        sx_bwd[1:] = 0.5 * (self.sx[:-1] + self.sx[1:])
        sx_bwd[0] = self.sx[0]

        sy_fwd = np.ones(Ny, dtype=complex)#same for y;
        sy_fwd[:-1] = 0.5 * (self.sy[:-1] + self.sy[1:])
        sy_fwd[-1] = self.sy[-1]

        sy_bwd = np.ones(Ny, dtype=complex)
        sy_bwd[1:] = 0.5 * (self.sy[:-1] + self.sy[1:])
        sy_bwd[0] = self.sy[0]

        dx2 = dx**2#shortcut;

        # Flatten indices: m = ix*Ny + iy;just for indexing
        ix_all = np.repeat(np.arange(Nx), Ny)  # (N,)
        iy_all = np.tile(np.arange(Ny), Nx)     # (N,)
        m_all = np.arange(N)                    # (N,)#m_all is the flattened index for each grid point, where m_all[i] corresponds to the grid point at (ix_all[i], iy_all[i]). 
        #example: if Nx=3 and Ny=4, then ix_all will be [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], iy_all will be [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], and m_all will be [0, 1, 2, ..., N-1]. 
        #so if i want to acces grid point (1,2), m=1*4+2=6. I have to call ix_all[6] and iy_all[6], m_all[6]=6, ix_all[6]=1, iy_all[6]=2.  
        # Stretched inverse factors for all grid points
        ISX_F = 1.0 / sx_fwd[ix_all]#just to store 1/stretch_factors.
        ISX_B = 1.0 / sx_bwd[ix_all]
        ISY_F = 1.0 / sy_fwd[iy_all]
        ISY_B = 1.0 / sy_bwd[iy_all]
        # a sparse matrix is a matrix that is mostly filled with zeros, and only a few non-zero entries. In the context of the FDFD method, the system matrix A is typically very large but also very sparse, because each grid point only interacts with its immediate neighbors 
        # ---- Build COO arrays ---- A COO array is a way to represent a sparse matrix by storing only the non-zero entries and their corresponding row and column indices. 
        rows_list = []
        cols_list = []
        vals_list = []

        # 1) Diagonal
        diag_val = -(ISX_F + ISX_B + ISY_F + ISY_B) / dx2 + k0**2 * self.eps_r.ravel()#ravel does the flattening self.eps_r[i]=
        rows_list.append(m_all)#m_all=[0, 1, 2, ..., N-1], 
        cols_list.append(m_all)#m_all=[0, 1, 2, ..., N-1], so this is the diagonal of the matrix A, where the row and column indices are the same (m_all), 
        vals_list.append(diag_val)
#diag_val is a 1D array of length N, where diag_val[i]= diagonal entry of matrix A corresponding to grid point (ix_all[i], iy_all[i]).  
        # 2) x+1 neighbor (m + Ny) -- only for ix < Nx-1
        mask_xp = ix_all < Nx - 1# 1D array of length N, mask_xp[1]=1 if the grid point (ix_all[1], iy_all[1]) has a neighbor in the +x direction 
        idx_src = m_all[mask_xp]#gives us the flattened indices of the grid points that have a neighbor in the +x direction, which are the source points for the x+1 neighbor interaction.
        idx_dst = idx_src + Ny#gives us the flattened indices of the neighboring grid points in the +x direction, which are the destination points for the x+1 neighbor interaction. Since each row corresponds to a grid point and each column corresponds to a grid point, adding Ny to the source index gives us the index of the neighbor in the +x direction.
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
# so by doing these, we add the contributions from the neighboring grid points in the +x, -x, +y, and -y directions to the sparse matrix A
        # Assemble
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        vals = np.concatenate(vals_list)

        A = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()#this is a function that build the sparse matrix A in COO format using the rows, cols, and vals arrays, and then converts it to CSC format for efficient solving. 
        b = -self.J_z.ravel()#this is the flattened version of the source term J_z

        return A, b

    def solve(self, verbose=True):#this function is called to build the system matrix A and the source vector b, and then solve the linear system A @ E_z = b for the electric field distribution E_z. The solution is reshaped back to a 2D array corresponding to the grid dimensions. The verbose flag allows for printing progress messages during the building and solving process.
        """Build and solve the FDFD system."""
        if verbose:
            print(f"Building FDFD system: {self.Nx}x{self.Ny} = {self.N} unknowns...")

        A, b = self.build_system()

        if verbose:
            print(f"Solving sparse system (nnz={A.nnz})...")

        self.E_z = spsolve(A, b).reshape(self.Nx, self.Ny)

        if verbose:
            print("Done.")

        return self.E_z#the result is a 2D array, where Ez[i][j]=the electric field at the grid point (i,j) in the computational domain. 

    def compute_H_fields(self):
        """Compute H_x, H_y from E_z using finite differences."""
        if self.E_z is None:
            raise ValueError("Solve first.")

        dx = self.dx
        k0 = self.k0

        # H_x = (1/(i*omega*mu)) * dE_z/dy
        # H_y = -(1/(i*omega*mu)) * dE_z/dx
        # With mu=1 and omega = k0: factor = 1/(i*k0)

        H_x = np.zeros_like(self.E_z)#2D array to store the H_x field, initialized to zeros. 
        H_y = np.zeros_like(self.E_z)#2D array to store the H_y field, initialized to zeros.

        # Central differences
        H_x[:, 1:-1] = (self.E_z[:, 2:] - self.E_z[:, :-2]) / (2 * dx * 1j * k0)
        H_y[1:-1, :] = -(self.E_z[2:, :] - self.E_z[:-2, :]) / (2 * dx * 1j * k0)
#aproximation of derivatives, Hx=dEz/dy, Hy=-dEz/dx
#dEz/dy at point (i,j) is approximated by (E_z[i, j+1] - E_z[i, j-1]) / (2*dx), and dE_z/dx at point (i,j) is approximated by (E_z[i+1, j] - E_z[i-1, j]) / (2*dx). The factors of 1/(i*k0) come from the relationship between the electric and magnetic fields in the frequency domain for TM polarization.
        return H_x, H_y

    def compute_poynting(self):
        """Compute time-averaged Poynting vector Re(E x H*/2)."""
        H_x, H_y = self.compute_H_fields()
        E = self.E_z

        # S = Re(E x H*) / 2
        # For TM mode: S_x = Re(E_z * H_y*)/2, S_y = -Re(E_z * H_x*)/2
        S_x = 0.5 * np.real(E * np.conj(H_y))
        S_y = -0.5 * np.real(E * np.conj(H_x))

        return S_x, S_y #S_x and S_y are 2D arrays representing the x and y components of the time-averaged Poynting vector at each grid point in the computational domain. 


# ============================================================================
# High-level simulation functions matching the paper's setup
# ============================================================================

def simulate_schwarzschild(b_inf, P_min=2.0, P0=6.0, n_annuli=16,
                            wavelength=0.5, M=2.5, resolution=15,
                            N_pml=12, verbose=True):
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
    N_pml : int
        Number of PML cells on each side.
    verbose : bool

    Returns
    -------
    fdfd : FDFD2D object with solved fields.
    """
    from ray_tracing import build_schwarzschild_annuli#returns the edges and n_values for the annular structure of the optical Schwarzschild black hole.

    # Physical dimensions
    R0 = P0 * M       # outer radius in um
    R_S = 2.0 * M     # Schwarzschild radius in um

    # Domain: 60 lambda x 60 lambda (paper specification)
    L = 60 * wavelength  # 30 um
    dx = wavelength / resolution

    if verbose:
        print(f"Schwarzschild FDFD: b_inf={b_inf}, M={M} um, lambda={wavelength} um")
        print(f"  Domain: {L:.1f} x {L:.1f} um, dx={dx:.4f} um, N_pml={N_pml}")

    fdfd = FDFD2D(L, L, dx, wavelength, N_pml)

    # Build annular system (dimensionless)
    edges_dim, n_values = build_schwarzschild_annuli(b_inf, P_min, P0, n_annuli)
    # Convert edges to physical units
    edges_phys = edges_dim * M

    # Set permittivity
    inner_eps = n_values[-1]**2 - 1j * np.pi#absorbption in the inner core.
    fdfd.set_annular_permittivity(0.0, 0.0, edges_phys, n_values, inner_eps)#returns eps_r, which is the 2D array of permittivity values for each grid.

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

    return fdfd#2D array of electric field values.


def simulate_kerr_newman(a, rho_Q, b_inf, ell_sign, P_min, P0=6.0,
                          n_annuli=21, wavelength=0.5, M=2.5,
                          resolution=15, N_pml=12, verbose=True):
    """
    Run FDFD simulation for an optical Kerr-Newman black hole.
    """
    from ray_tracing import build_kn_annuli#returns the edges and n_values for the annular structure of the optical Kerr-Newman black hole.

    R0 = P0 * M
    L = 60 * wavelength
    dx = wavelength / resolution

    if verbose:
        print(f"Kerr-Newman FDFD: a={a}, rho_Q={rho_Q}, b_inf={b_inf}")
        print(f"  Domain: {L:.1f} x {L:.1f} um, dx={dx:.4f} um, "
              f"N_pml={N_pml}, P_min={P_min:.3f}")

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

    fdfd.solve(verbose=verbose)#maybe remove the verbose or something
    return fdfd