import numpy as np
from scipy.spatial import cKDTree

from clusterinnout.distance_utils import periodic_allclose, periodic_distance


class FilamentReader:
    def __init__(self, skl_crits_file, skl_segs_file, box_size):
        self._box_size = box_size

        self.node_pos, self.void_pos = self._read_skl_crits(skl_crits_file)

        self.filament_arcs = self._read_skl_segs(skl_segs_file)

    def _read_skl_crits(self, skl_crits_file):
        _crits_data = np.loadtxt(skl_crits_file)

        _node_mask = np.logical_and(_crits_data[:, 6] == 0, _crits_data[:, 4] == 3)
        _void_mask = np.logical_and(_crits_data[:, 6] == 0, _crits_data[:, 4] == 0)

        node_pos = _crits_data[_node_mask, 0:3] % self._box_size
        void_pos = _crits_data[_void_mask, 0:3] % self._box_size

        return node_pos, void_pos

    def _read_skl_segs(self, skl_segs_file):
        _segs_data = np.loadtxt(skl_segs_file)

        _mask = np.logical_and(
            _segs_data[:, 8] == 2,
            _segs_data[:, 9] == 0,
        )

        _seg_u_pos = _segs_data[_mask, 0:3] % self._box_size
        _seg_v_pos = _segs_data[_mask, 3:6] % self._box_size

        filament_arcs = []

        _current = [_seg_u_pos[0], _seg_v_pos[0]]
        for i in range(1, _seg_u_pos.shape[0]):
            if periodic_allclose(_seg_u_pos[i], _seg_v_pos[i], self._box_size):
                continue

            if periodic_allclose(_seg_u_pos[i], _seg_v_pos[i - 1], self._box_size):
                _current.append(_seg_v_pos[i])
            else:
                filament_arcs.append(np.array(_current))
                _current = [_seg_u_pos[i], _seg_v_pos[i]]
        filament_arcs.append(np.array(_current))

        return filament_arcs

    def mask_nodes(self, cluster_pos, threshold=1e3):
        _cluster_tree = cKDTree(cluster_pos, boxsize=self._box_size)

        _node_cluster_dist, _ = _cluster_tree.query(self.node_pos)

        self.false_node_pos = self.node_pos[_node_cluster_dist >= threshold]
        self.node_pos = self.node_pos[_node_cluster_dist < threshold]

    def _get_arc_ends(self):
        return np.concatenate(
            [arc[-1].reshape(-1, 3) for arc in self.filament_arcs],
            axis=0,
        )

    def _remove_false_filament_arcs(self, false_filament_indices):
        self.filament_arcs = [
            self.filament_arcs[i]
            for i in range(len(self.filament_arcs))
            if i not in false_filament_indices
        ]

    def mask_seg_arcs(self):
        if not hasattr(self, "false_node_pos"):
            _msg = "call mask_nodes() before mask_seg_arcs()"
            raise RuntimeError(_msg)

        _arc_ends = self._get_arc_ends()
        _arc_end_tree = cKDTree(_arc_ends, boxsize=self._box_size)

        _false_node_arc_end_dist, _false_node_arc_end_idx = _arc_end_tree.query(
            self.false_node_pos
        )

        _false_filament_indices = _false_node_arc_end_idx[
            _false_node_arc_end_dist < 1e-1
        ]

        self._remove_false_filament_arcs(_false_filament_indices)

    def get_filament_cluster_indices_and_lengths(
        self, cluster_pos, cluster_radius, n_ngb=8, thresholds=(1, 3)
    ):
        _cluster_tree = cKDTree(cluster_pos, boxsize=self._box_size)

        _arc_ends = self._get_arc_ends()
        _arc_end_cluster_dist, _arc_end_cluster_idx = _cluster_tree.query(
            _arc_ends, k=n_ngb
        )

        _arc_end_cluster_norm_dist = (
            _arc_end_cluster_dist / cluster_radius[_arc_end_cluster_idx]
        )

        filament_cluster_indices = []
        filament_lengths = []
        filament_effective_lengths = []

        _false_filament_indices = []

        for i, _arc in enumerate(self.filament_arcs):
            _arc_end_norm_dist = _arc_end_cluster_norm_dist[i].reshape(-1)
            _arc_end_idx = _arc_end_cluster_idx[i].reshape(-1)

            _filament_cluster_idx = _arc_end_idx[np.argmin(_arc_end_norm_dist)]

            _filament_cluster_distance = periodic_distance(
                _arc, cluster_pos[_filament_cluster_idx], self._box_size
            )
            if (
                _filament_cluster_distance[-1]
                > thresholds[0] * cluster_radius[_filament_cluster_idx]
            ):
                _false_filament_indices.append(i)
                continue

            _seg_lengths = periodic_distance(_arc[:-1], _arc[1:], self._box_size)

            _filament_length = np.sum(_seg_lengths)
            _filament_effective_length = np.sum(
                _seg_lengths[
                    _filament_cluster_distance[:-1]
                    > thresholds[1] * cluster_radius[_filament_cluster_idx]
                ]
            )

            if _filament_effective_length < 1e-1:
                _false_filament_indices.append(i)
                continue

            filament_cluster_indices.append(_filament_cluster_idx)
            filament_lengths.append(_filament_length)
            filament_effective_lengths.append(_filament_effective_length)

        self._remove_false_filament_arcs(_false_filament_indices)

        return (
            np.array(filament_cluster_indices),
            np.array(filament_lengths),
            np.array(filament_effective_lengths),
        )

    def save_filament_arcs(self, filament_arcs_file):
        np.savez(
            filament_arcs_file,
            **{f"{i}": self.filament_arcs[i] for i in range(len(self.filament_arcs))},
        )


class WallReader:
    def __init__(self, wall_file):
        self.wall_triangles = self._read_wall_triangles(wall_file)

    def _read_wall_triangles(self, wall_file):
        with open(wall_file) as f:  # noqa: PTH123
            lines = [l.strip() for l in f]

        i = 2

        while lines[i].startswith("#") or lines[i].startswith("BBOX"):
            i += 1

        nv = int(lines[i])
        i += 1

        vertices = np.zeros((nv, 3), dtype=float)
        for j in range(nv):
            coords = np.array(lines[i].split(), dtype=float)
            vertices[j] = coords
            i += 1

        wall_triangle_indices = []
        nw = int(lines[i].split()[1])
        i += 1

        for _ in range(nw):
            idx = tuple(map(int, lines[i].split()))
            wall_triangle_indices.append(idx)
            i += 1

        return np.array(vertices)[wall_triangle_indices]

    def save_wall_triangles(self, wall_triangles_file):
        np.save(wall_triangles_file, self.wall_triangles)
