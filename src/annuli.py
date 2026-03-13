import numpy as np


def annulus_edges_with_half_ends(P_min, P_max, n_annuli):
    """
    Outer and inner annuli have half the width of each interior annulus.
    Returns edges ordered from outer to inner.
    """
    if n_annuli < 2:
        raise ValueError("n_annuli must be at least 2")

    total_width = P_max - P_min
    interior_width = total_width / (n_annuli - 1)

    widths = [interior_width / 2.0] + [interior_width] * (n_annuli - 2) + [interior_width / 2.0]

    edges = [P_max]
    current = P_max
    for w in widths:
        current -= w
        edges.append(current)

    return np.array(edges, dtype=float)


def annulus_centers_from_edges(edges):
    return 0.5 * (edges[:-1] + edges[1:])


def sample_piecewise_constant(index_function, edges, *args, **kwargs):
    """
    Paper rule:
    - outermost value = n(P_max)
    - innermost value = n(P_min)
    - interior annuli use n(center)
    """
    centers = annulus_centers_from_edges(edges)
    values = index_function(centers, *args, **kwargs)

    # Replace outermost / innermost by endpoint values, as stated in the paper
    values = np.asarray(values, dtype=float)
    values[0] = index_function(np.array([edges[0]]), *args, **kwargs)[0]      # P_max
    values[-1] = index_function(np.array([edges[-1]]), *args, **kwargs)[0]    # P_min
    return centers, values