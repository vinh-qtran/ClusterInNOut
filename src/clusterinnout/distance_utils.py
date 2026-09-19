import numpy as np


def periodic_difference(pos1, pos2, box_size):
    _delta = pos1 - pos2
    _delta = np.where(_delta > 0.5 * box_size, _delta - box_size, _delta)
    return np.where(_delta < -0.5 * box_size, _delta + box_size, _delta)


def periodic_allclose(pos1, pos2, box_size, atol=1):
    _delta = periodic_difference(pos1, pos2, box_size)
    return np.all(np.abs(_delta) < atol, axis=-1)


def periodic_distance(pos1, pos2, box_size):
    _delta = periodic_difference(pos1, pos2, box_size)
    return np.linalg.norm(_delta, axis=-1)


def periodic_mean(pos, box_size, axis=0):
    _ref = np.take(pos, 0, axis=axis)
    _ref = np.expand_dims(_ref, axis=axis)
    _unwrapped = _ref + periodic_difference(pos, _ref, box_size)
    return np.mean(_unwrapped, axis=axis) % box_size


def periodic_length(pos, box_size):
    _delta = periodic_difference(pos[1:], pos[:-1], box_size)
    return np.sum(np.linalg.norm(_delta, axis=-1))


def periodic_interpolate(pos1, pos2, box_size, n_interp=15):
    _delta = periodic_difference(pos2, pos1, box_size)
    return np.concatenate(
        [
            (pos1 + (i / (n_interp + 1)) * _delta) % box_size
            for i in range(n_interp + 2)
        ],
        axis=0,
    )


def periodic_point_line_distance(p, a, b, box_size, atol=1):
    _ab = periodic_difference(b, a, box_size)
    _ap = periodic_difference(p, a, box_size)

    _line_length = np.linalg.norm(_ab, axis=-1)
    _safe_length = np.where(_line_length < atol, 1.0, _line_length)
    _line_unit_vec = _ab / _safe_length[..., None]

    _proj_length = np.sum(_ap * _line_unit_vec, axis=-1)
    _proj_length_clamped = np.clip(_proj_length, 0.0, _line_length)

    _proj_point = (a + _proj_length_clamped[..., None] * _line_unit_vec) % box_size

    _dist = np.linalg.norm(periodic_difference(p, _proj_point, box_size), axis=-1)

    return np.where(_line_length < atol, np.linalg.norm(_ap, axis=-1), _dist)


def _closest_point_on_triangle(p, a, b, c, atol=1):
    _ab = b - a
    _ac = c - a
    _ap = p - a

    _d1 = np.sum(_ab * _ap, axis=-1)
    _d2 = np.sum(_ac * _ap, axis=-1)
    _case_a = (_d1 <= 0) & (_d2 <= 0)

    _bp = p - b
    _d3 = np.sum(_ab * _bp, axis=-1)
    _d4 = np.sum(_ac * _bp, axis=-1)
    _case_b = (_d3 >= 0) & (_d4 <= _d3) & ~_case_a

    _vc = _d1 * _d4 - _d3 * _d2
    _case_ab = (_vc <= 0) & (_d1 >= 0) & (_d3 <= 0) & ~_case_a & ~_case_b

    _cp = p - c
    _d5 = np.sum(_ab * _cp, axis=-1)
    _d6 = np.sum(_ac * _cp, axis=-1)
    _case_c = (_d6 >= 0) & (_d5 <= _d6) & ~_case_a & ~_case_b & ~_case_ab

    _vb = _d5 * _d2 - _d1 * _d6
    _case_ac = (
        (_vb <= 0)
        & (_d2 >= 0)
        & (_d6 <= 0)
        & ~_case_a
        & ~_case_b
        & ~_case_ab
        & ~_case_c
    )

    _va = _d3 * _d6 - _d5 * _d4
    _case_bc = (
        (_va <= 0)
        & ((_d4 - _d3) >= 0)
        & ((_d5 - _d6) >= 0)
        & ~_case_a
        & ~_case_b
        & ~_case_ab
        & ~_case_c
        & ~_case_ac
    )

    _case_face = ~(_case_a | _case_b | _case_ab | _case_c | _case_ac | _case_bc)

    _denom_ab = _d1 - _d3
    _v_ab = np.divide(
        _d1, _denom_ab, out=np.zeros_like(_d1), where=np.abs(_denom_ab) > atol
    )

    _denom_ac = _d2 - _d6
    _w_ac = np.divide(
        _d2, _denom_ac, out=np.zeros_like(_d2), where=np.abs(_denom_ac) > atol
    )

    _denom_bc = (_d4 - _d3) + (_d5 - _d6)
    _w_bc = np.divide(
        _d4 - _d3, _denom_bc, out=np.zeros_like(_d4), where=np.abs(_denom_bc) > atol
    )

    _denom_face = _va + _vb + _vc
    _v_face = np.divide(
        _vb, _denom_face, out=np.zeros_like(_vb), where=np.abs(_denom_face) > atol
    )
    _w_face = np.divide(
        _vc, _denom_face, out=np.zeros_like(_vc), where=np.abs(_denom_face) > atol
    )

    result = np.zeros_like(a)
    result[_case_a] = a[_case_a]
    result[_case_b] = b[_case_b]
    result[_case_c] = c[_case_c]
    result[_case_ab] = (a + _v_ab[..., None] * _ab)[_case_ab]
    result[_case_ac] = (a + _w_ac[..., None] * _ac)[_case_ac]
    result[_case_bc] = (b + _w_bc[..., None] * (c - b))[_case_bc]
    result[_case_face] = (a + _v_face[..., None] * _ab + _w_face[..., None] * _ac)[
        _case_face
    ]

    return result


def periodic_point_triangle_distance(p, a, b, c, box_size):
    _a_local = periodic_difference(a, p, box_size)
    _b_local = periodic_difference(b, p, box_size)
    _c_local = periodic_difference(c, p, box_size)
    _p_local = np.zeros_like(_a_local)

    _closest_local = _closest_point_on_triangle(_p_local, _a_local, _b_local, _c_local)

    return np.linalg.norm(_closest_local - _p_local, axis=-1)
