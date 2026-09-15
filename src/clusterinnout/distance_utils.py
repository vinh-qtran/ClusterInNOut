import numpy as np


def periodic_difference(pos1, pos2, box_size):
    _delta = pos1 - pos2
    _delta = np.where(_delta > 0.5 * box_size, _delta - box_size, _delta)
    return np.where(_delta < -0.5 * box_size, _delta + box_size, _delta)


def periodic_allclose(pos1, pos2, box_size, atol=1e-1):
    _delta = periodic_difference(pos1, pos2, box_size)
    return np.all(np.abs(_delta) < atol, axis=-1)


def periodic_distance(pos1, pos2, box_size):
    _delta = periodic_difference(pos1, pos2, box_size)
    return np.sqrt((_delta**2).sum(axis=-1))


def periodic_interpolate(pos1, pos2, box_size, n_interp=100):
    _delta = periodic_difference(pos2, pos1, box_size)
    return np.concatenate(
        [(pos1 + (i / n_interp) * _delta) % box_size for i in range(1, n_interp + 1)],
        axis=0,
    )


def interp_filament_arcs(filament_arcs, box_size, n_interp=100):
    arc_midpoints = np.concatenate(
        [
            periodic_interpolate(arc[:-1], arc[1:], box_size, n_interp=n_interp)
            for arc in filament_arcs
        ],
        axis=0,
    )

    arc_counts = np.array([n_interp * (arc.shape[0] - 1) for arc in filament_arcs])

    return arc_midpoints, arc_counts
