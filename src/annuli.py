import numpy as np


def annulus_edges_with_half_ends(P_min, P_max, n_annuli):
    """
    Outer and inner annuli have half the width of each interior annulus.
    Returns edges ordered from outer to inner.

    Special case: for a single annulus, the system is just [P_max, P_min].
    """
    if n_annuli < 1:
        raise ValueError("n_annuli must be at least 1")

    if n_annuli == 1:
        return np.array([P_max, P_min], dtype=float) # Special case: single annulus with half-width ends is just the whole range

    total_width = P_max - P_min
    interior_width = total_width / (n_annuli - 1)

    widths = [interior_width / 2.0] + [interior_width] * (n_annuli - 2) + [interior_width / 2.0]
# The widths list defines the width of each annulus, with the first and last annuli having half the width of the interior annuli. The total width is divided by (n_annuli - 1) to determine the width of the interior annuli, and then the first and last widths are set to half of that value.
    edges = [P_max]# Start with the outer edge at P_max
    current = P_max
    for w in widths:# Iterate through the widths and calculate the edges by subtracting the width from the current edge. This creates the edges of the annuli ordered from outer to inner.
        current -= w
        edges.append(current)

    return np.array(edges, dtype=float)


def annulus_centers_from_edges(edges):
    return 0.5 * (edges[:-1] + edges[1:])# The centers of the annuli are calculated as the average of the edges of each annulus. This is done by taking the edges array, slicing it to get the outer and inner edges of each annulus, and then averaging them to find the center.
#example: if edges = [3, 2, 1], then edges[:-1] = [3, 2] and edges[1:] = [2, 1]. The centers would be calculated as 0.5 * ([3, 2] + [2, 1]) = 0.5 * ([5, 3]) = [2.5, 1.5], which are the centers of the annuli defined by the edges.

def sample_piecewise_constant(index_function, edges, *args, **kwargs):# This function samples the index_function at the centers of the annuli defined by the edges. It takes the index_function, the edges of the annuli, and any additional arguments or keyword arguments needed to evaluate the index_function.
    """
    Paper rule:
    - outermost value = n(P_max)
    - innermost value = n(P_min)
    - interior annuli use n(center)
    """
    centers = annulus_centers_from_edges(edges)# this are the annuli centers.
    values = index_function(centers, *args, **kwargs)# The index_function is evaluated at the centers of the annuli to get the refractive index values for each annulus. The additional arguments and keyword arguments are passed to the index_function as needed.
#we evaluate n at the center of each annuli, but we will replace the outermost and innermost values with n(P_max) and n(P_min) respectively, as per the paper's rule.
    # Replace outermost / innermost by endpoint values, as stated in the paper
    values = np.asarray(values, dtype=float)#transform into array, for computational purposes.
    values[0] = index_function(np.array([edges[0]]), *args, **kwargs)[0]   # P_max
    if len(values) > 1:
        values[-1] = index_function(np.array([edges[-1]]), *args, **kwargs)[0]  # P_min
    return centers, values#we retunr two arrays: the centers of each annuli and the corresponding values of the refractive index. note that we may use whichever index_function here depending on the case: Schwarzschild or Kerr-Newman.
