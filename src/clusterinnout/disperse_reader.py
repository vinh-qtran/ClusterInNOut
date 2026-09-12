import numpy as np
from scipy.spatial import cKDTree


class FilamentReader:
    def __init__(self, skl_crits_file, skl_segs_file, box_size):
        self._box_size = box_size

        self.node_pos, self.void_pos = self._read_skl_crits(skl_crits_file)

        self.seg_arcs = self._read_skl_segs(skl_segs_file)

    def _read_skl_crits(self, skl_crits_file):
        _crits_data = np.loadtxt(skl_crits_file)

        _node_mask = np.logical_and(_crits_data[:, 6] == 0, _crits_data[:, 4] == 3)
        _void_mask = np.logical_and(_crits_data[:, 6] == 0, _crits_data[:, 4] == 0)

        node_pos = _crits_data[_node_mask, 0:3]
        void_pos = _crits_data[_void_mask, 0:3]

        return node_pos, void_pos

    def _read_skl_segs(self, skl_segs_file):
        _segs_data = np.loadtxt(skl_segs_file)

        _mask = _segs_data[:, 9] == 0

        _seg_u_pos = _segs_data[_mask, 0:3]
        _seg_v_pos = _segs_data[_mask, 3:6]

        seg_arcs = []

        _current = [_seg_u_pos[0], _seg_v_pos[0]]
        for i in range(1, _seg_u_pos.shape[0]):
            if np.allclose(_seg_u_pos[i], _seg_v_pos[i]):
                continue

            if np.allclose(_seg_u_pos[i], _seg_v_pos[i - 1]):
                _current.append(_seg_v_pos[i])
            else:
                seg_arcs.append(np.array(_current))
                _current = [_seg_u_pos[i], _seg_v_pos[i]]
        seg_arcs.append(np.array(_current))

        return seg_arcs

    def mask_seg_arcs(self, cluster_pos, thres=1e3):
        _cluster_tree = cKDTree(cluster_pos, boxsize=self._box_size)

        _cluster_dist, _ = _cluster_tree.query(self.node_pos)

        _false_node_pos = self.node_pos[_cluster_dist >= thres]

        _arc_ends = np.concatenate(self.seg_arcs, axis=0)
        _arc_lengths = np.array([arc.shape[0] for arc in self.seg_arcs])

        _filament_tree = cKDTree(_arc_ends, boxsize=self._box_size)

        _false_filament_dist, _false_filament_end_idx = _filament_tree.query(
            _false_node_pos
        )

        _false_filament_indices = np.searchsorted(
            np.cumsum(_arc_lengths), _false_filament_end_idx, side="right"
        )

        _false_filament_indices = _false_filament_indices[_false_filament_dist == 0]

        self.seg_arcs = [
            self.seg_arcs[i]
            for i in range(len(self.seg_arcs))
            if i not in _false_filament_indices
        ]

    def save_filament_arcs(self, filament_arcs_file):
        np.savez(
            filament_arcs_file,
            **{f"{i}": self.seg_arcs[i] for i in range(len(self.seg_arcs))},
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
