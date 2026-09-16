import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from clusterinnout.distance_utils import (
    periodic_allclose,
    periodic_difference,
    periodic_distance,
    periodic_interpolate,
    periodic_mean,
    periodic_point_triangle_distance,
)


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

    def mask_nodes(self, cluster_pos, cluster_radius, n_candidates=32, threshold=1):
        _cluster_tree = cKDTree(cluster_pos, boxsize=self._box_size)

        _node_cluster_dististances, _node_cluster_indices = _cluster_tree.query(
            self.node_pos, k=n_candidates
        )
        _node_cluster_norm_distances = (
            _node_cluster_dististances / cluster_radius[_node_cluster_indices]
        )

        _node_idx = np.arange(_node_cluster_indices.shape[0])
        _candidate_idx = np.argmin(_node_cluster_norm_distances, axis=-1)

        _node_cluster_norm_dist = _node_cluster_norm_distances[
            _node_idx, _candidate_idx
        ]

        self.false_node_pos = self.node_pos[_node_cluster_norm_dist >= threshold]
        self.node_pos = self.node_pos[_node_cluster_norm_dist < threshold]

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

    def mask_seg_arcs(self, atol=1e-2):
        if not hasattr(self, "false_node_pos"):
            _msg = "call mask_nodes() before mask_seg_arcs()"
            raise RuntimeError(_msg)

        _arc_ends = self._get_arc_ends()

        _false_node_tree = cKDTree(self.false_node_pos, boxsize=self._box_size)

        _arc_end_false_node_dist, _ = _false_node_tree.query(_arc_ends)

        self._remove_false_filament_arcs(np.where(_arc_end_false_node_dist < atol)[0])

    def get_filament_cluster_indices_and_lengths(
        self,
        cluster_pos,
        cluster_radius,
        n_candidates=32,
        thresholds=(0.5, 2),
        atol=1e-2,
    ):
        _cluster_tree = cKDTree(cluster_pos, boxsize=self._box_size)

        _arc_ends = self._get_arc_ends()
        _arc_end_cluster_distances, _arc_end_cluster_indices = _cluster_tree.query(
            _arc_ends, k=n_candidates
        )

        _arc_end_cluster_norm_distances = (
            _arc_end_cluster_distances / cluster_radius[_arc_end_cluster_indices]
        )

        filament_cluster_indices = []
        filament_lengths = []
        filament_effective_lengths = []

        _false_filament_indices = []

        for i, _arc in tqdm(
            enumerate(self.filament_arcs), total=len(self.filament_arcs)
        ):
            _arc_end_norm_distances = _arc_end_cluster_norm_distances[i].reshape(-1)
            _arc_end_indices = _arc_end_cluster_indices[i].reshape(-1)

            _filament_cluster_idx = _arc_end_indices[np.argmin(_arc_end_norm_distances)]

            _filament_cluster_distance = periodic_distance(
                _arc, cluster_pos[_filament_cluster_idx], self._box_size
            )
            _filament_cluster_norm_distance = (
                _filament_cluster_distance / cluster_radius[_filament_cluster_idx]
            )

            if _filament_cluster_norm_distance[-1] > thresholds[0]:
                _false_filament_indices.append(i)
                continue

            _seg_lengths = periodic_distance(_arc[:-1], _arc[1:], self._box_size)

            _filament_length = np.sum(_seg_lengths)
            _filament_effective_length = np.sum(
                _seg_lengths[_filament_cluster_norm_distance[:-1] > thresholds[1]]
            )

            if _filament_effective_length < atol:
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

    def _get_segment_arrays(self):
        _seg_starts, _seg_ends, _seg_arc_idx = [], [], []
        for i, arc in enumerate(self.filament_arcs):
            _seg_starts.append(arc[:-1])
            _seg_ends.append(arc[1:])
            _seg_arc_idx.append(np.full(arc.shape[0] - 1, i))

        return (
            np.concatenate(_seg_starts, axis=0),
            np.concatenate(_seg_ends, axis=0),
            np.concatenate(_seg_arc_idx, axis=0),
        )

    def _get_galaxy_segment_distances(
        self, galaxy_pos, line_start, line_end, box_size, atol=1e-2
    ):
        _galaxy_pos = galaxy_pos.reshape(-1, 1, 3)

        _line_vec = periodic_difference(line_end, line_start, box_size)
        _point_vec = periodic_difference(_galaxy_pos, line_start, box_size)

        _line_length = np.linalg.norm(_line_vec, axis=-1)
        _safe_length = np.where(_line_length < atol, 1.0, _line_length)
        _line_unit_vec = _line_vec / _safe_length[..., None]

        _proj_length = np.sum(_point_vec * _line_unit_vec, axis=-1)
        _proj_length_clamped = np.clip(_proj_length, 0.0, _line_length)

        _proj_point = (
            line_start + _proj_length_clamped[..., None] * _line_unit_vec
        ) % box_size

        _dist = np.linalg.norm(
            periodic_difference(_galaxy_pos, _proj_point, box_size), axis=-1
        )

        return np.where(_line_length < atol, np.linalg.norm(_point_vec, axis=-1), _dist)

    def get_galaxy_filament_distances(self, galaxy_pos, n_interp=15, n_candidates=32):
        _seg_starts, _seg_ends, _seg_arc_idx = self._get_segment_arrays()

        _interp_points = periodic_interpolate(
            _seg_starts, _seg_ends, self._box_size, n_interp=n_interp
        )
        _interp_point_seg_indices = np.tile(
            np.arange(_seg_starts.shape[0]), n_interp + 2
        )

        _filament_tree = cKDTree(_interp_points, boxsize=self._box_size)
        _, _candidate_interp_point_indices = _filament_tree.query(
            galaxy_pos, k=n_candidates
        )

        _candidate_seg_indices = _interp_point_seg_indices[
            _candidate_interp_point_indices
        ]

        _candidate_seg_starts = _seg_starts[_candidate_seg_indices]
        _candidate_seg_ends = _seg_ends[_candidate_seg_indices]

        _galaxy_seg_distances = self._get_galaxy_segment_distances(
            galaxy_pos, _candidate_seg_starts, _candidate_seg_ends, self._box_size
        )

        _best_candidate_indices = np.argmin(_galaxy_seg_distances, axis=-1)
        _galalaxy_indices = np.arange(galaxy_pos.shape[0])

        galaxy_filament_distances = _galaxy_seg_distances[
            _galalaxy_indices, _best_candidate_indices
        ]
        galaxy_filament_indices = _seg_arc_idx[
            _candidate_seg_indices[_galalaxy_indices, _best_candidate_indices]
        ]

        return galaxy_filament_distances, galaxy_filament_indices

    def get_filament_effective_masses(
        self,
        filament_cluster_pos,
        filament_cluster_radius,
        galaxy_pos,
        galaxy_masses,
        galaxy_filament_distances,
        galaxy_filament_indices,
        r_filament=1e3,
        threshold=1,
    ):
        _galaxy_mask = np.logical_and(
            galaxy_filament_distances < r_filament,
            periodic_distance(
                galaxy_pos,
                filament_cluster_pos[galaxy_filament_indices],
                self._box_size,
            )
            / filament_cluster_radius[galaxy_filament_indices]
            > threshold,
        )

        _galaxy_mass = galaxy_masses[_galaxy_mask]
        _galaxy_filament_indices = galaxy_filament_indices[_galaxy_mask]

        return np.array(
            [
                np.sum(_galaxy_mass[_galaxy_filament_indices == i])
                for i in range(len(self.filament_arcs))
            ]
        )

    def save_filament_arcs(self, filament_arcs_file):
        np.savez(
            filament_arcs_file,
            **{f"{i}": self.filament_arcs[i] for i in range(len(self.filament_arcs))},
        )


class WallReader:
    def __init__(self, wall_file, box_size):
        self._box_size = box_size

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
            vertices[j] = np.array(lines[i].split(), dtype=float)
            i += 1

        triangle_indices = None
        while (
            i < len(lines) and lines[i] and not lines[i].startswith("[ADDITIONAL_DATA]")
        ):
            header = lines[i].split()
            if len(header) != 2 or not header[0].lstrip("-").isdigit():
                break  # not a "T N" line -> end of simplex blocks
            t_dim, n_simplices = int(header[0]), int(header[1])
            i += 1

            block = [list(map(int, lines[i + k].split())) for k in range(n_simplices)]
            i += n_simplices

            if t_dim == 2:
                triangle_indices = np.array(block)

        if triangle_indices is None:
            _msg = f"No 2-simplex (triangle) block found in {wall_file}."
            raise ValueError(_msg)

        return vertices[triangle_indices] % self._box_size

    def get_galaxy_wall_distances(
        self, galaxy_pos, dense_midpoints=True, n_candidates=32
    ):
        _n_tri = self.wall_triangles.shape[0]

        _midpoints = np.concatenate(
            [
                self.wall_triangles.reshape(-1, 3),
                periodic_mean(self.wall_triangles, box_size=self._box_size, axis=1),
                periodic_mean(
                    self.wall_triangles[:, [0, 1]], box_size=self._box_size, axis=1
                ),
                periodic_mean(
                    self.wall_triangles[:, [1, 2]], box_size=self._box_size, axis=1
                ),
                periodic_mean(
                    self.wall_triangles[:, [2, 0]], box_size=self._box_size, axis=1
                ),
            ]
            + (
                [
                    periodic_mean(
                        self.wall_triangles[:, [0, 0, 0, 1]],
                        box_size=self._box_size,
                        axis=1,
                    ),
                    periodic_mean(
                        self.wall_triangles[:, [1, 1, 1, 2]],
                        box_size=self._box_size,
                        axis=1,
                    ),
                    periodic_mean(
                        self.wall_triangles[:, [2, 2, 2, 0]],
                        box_size=self._box_size,
                        axis=1,
                    ),
                    periodic_mean(
                        self.wall_triangles[:, [0, 1, 1, 1]],
                        box_size=self._box_size,
                        axis=1,
                    ),
                    periodic_mean(
                        self.wall_triangles[:, [1, 2, 2, 2]],
                        box_size=self._box_size,
                        axis=1,
                    ),
                    periodic_mean(
                        self.wall_triangles[:, [2, 0, 0, 0]],
                        box_size=self._box_size,
                        axis=1,
                    ),
                    periodic_mean(
                        self.wall_triangles[:, [0, 1, 2, 0]],
                        box_size=self._box_size,
                        axis=1,
                    ),
                    periodic_mean(
                        self.wall_triangles[:, [1, 2, 0, 1]],
                        box_size=self._box_size,
                        axis=1,
                    ),
                    periodic_mean(
                        self.wall_triangles[:, [2, 0, 1, 2]],
                        box_size=self._box_size,
                        axis=1,
                    ),
                ]
                if dense_midpoints
                else []
            ),
            axis=0,
        )

        _midpoint_indices = np.concatenate(
            [
                np.repeat(np.arange(_n_tri), 3),
                np.tile(np.arange(_n_tri), 13 if dense_midpoints else 4),
            ]
        )

        _wall_tree = cKDTree(_midpoints, boxsize=self._box_size)
        _, _candidate_midpoint_indices = _wall_tree.query(galaxy_pos, k=n_candidates)

        _candidate_triangle_idx = _midpoint_indices[_candidate_midpoint_indices]

        _candidate_a = self.wall_triangles[_candidate_triangle_idx, 0]
        _candidate_b = self.wall_triangles[_candidate_triangle_idx, 1]
        _candidate_c = self.wall_triangles[_candidate_triangle_idx, 2]

        _galaxy_pos = galaxy_pos.reshape(-1, 1, 3)

        _candidate_distances = periodic_point_triangle_distance(
            _galaxy_pos, _candidate_a, _candidate_b, _candidate_c, self._box_size
        )

        return np.min(_candidate_distances, axis=-1)

    def save_wall_triangles(self, wall_triangles_file):
        np.save(wall_triangles_file, self.wall_triangles)
