import json

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from clusterinnout.distance_utils import (
    periodic_cluster_match,
    periodic_distance,
    periodic_interpolate,
    periodic_length,
    periodic_mean,
    periodic_point_line_distance,
    periodic_point_triangle_distance,
)


class FilamentReader:
    def __init__(
        self, ndskl_file, box_size, cluster_dict=None, galaxy_dict=None, r_filament=1e3
    ):
        self._box_size = box_size
        self._r_filament = r_filament

        self._read_ndskl_ascii(ndskl_file)
        self._assign_nodes()

        if cluster_dict is not None:
            self.ClusterNode_Idx, self.Node_ClusterIdx = self._match_node_to_cluster(
                self.Node_Pos,
                cluster_dict["Position"],
                cluster_dict["Rsp"],
            )

            self.ClusterFilament_Idx = np.nonzero(
                np.isin(self.Filament_NodeIdx, self.ClusterNode_Idx)
            )[0]

            _valid = np.isin(self.Filament_NodeIdx, self.ClusterNode_Idx)
            self.Filament_ClusterIdx = np.where(
                _valid, self.Node_ClusterIdx[self.Filament_NodeIdx], -1
            )

            self.Filament_EffLength = self._get_filament_eff_length(
                self.Filament_Arc,
                cluster_dict["Position"],
                cluster_dict["Rsp"],
            )

            if galaxy_dict is not None:
                self.Galaxy_FilamentDistance, self.Galaxy_FilamentIdx = (
                    self._get_galaxy_filament_distance(
                        self.Filament_Arc, galaxy_dict["Position"]
                    )
                )

                self.Filament_EffMass = self._get_filament_effective_mass(
                    len(self.Filament_Arc),
                    galaxy_dict["Mass"],
                    galaxy_dict["ClusterNormDistance"],
                    self.Galaxy_FilamentDistance,
                    self.Galaxy_FilamentIdx,
                    r_filament,
                )

                self.Filament_EffMassPerLength = np.where(
                    self.Filament_EffLength > 0,
                    self.Filament_EffMass / self.Filament_EffLength,
                    0.0,
                )

    def _read_ndskl_ascii(self, filename, allowed_kinds=None):
        if allowed_kinds is None:
            allowed_kinds = {(3, 2), (3, 4), (4, 2), (4, 4)}

        with open(filename) as f:  # noqa: PTH123
            _lines = [_ln.strip() for _ln in f if _ln.strip()]

        _ndims = int(_lines[1].split()[0])

        i = _lines.index("[CRITICAL POINTS]") + 1
        _ncrit = int(_lines[i])
        i += 1
        _cp_type = np.empty(_ncrit, int)
        _cp_pos = np.empty((_ncrit, _ndims))
        _cp_bnd = np.empty(_ncrit, int)
        for _k in range(_ncrit):
            _v = _lines[i].split()
            _cp_type[_k] = int(_v[0])
            _cp_pos[_k] = np.array(_v[1 : 1 + _ndims], float)
            _cp_bnd[_k] = int(_v[3 + _ndims])
            i += 2 + int(_lines[i + 1])

        i = _lines.index("[FILAMENTS]", i) + 1
        _nfil = int(_lines[i])
        i += 1
        _fil_cp = np.empty((_nfil, 2), int)
        _fil_pts = []
        for _k in range(_nfil):
            _c1, _c2, _n = map(int, _lines[i].split())
            i += 1
            _fil_cp[_k] = _c1, _c2
            _fil_pts.append(
                np.array(" ".join(_lines[i : i + _n]).split(), float).reshape(
                    _n, _ndims
                )
            )
            i += _n

        _keep_cp = np.isin(_cp_type, [2, 3, 4]) & (_cp_bnd == 0)
        _new_id = np.full(_ncrit, -1)
        _new_id[_keep_cp] = np.arange(_keep_cp.sum())

        _t = _cp_type[_fil_cp]
        _pair_ok = np.array(
            [(a, b) in allowed_kinds or (b, a) in allowed_kinds for a, b in _t], bool
        ).reshape(-1)
        _keep_fil = np.nonzero(_keep_cp[_fil_cp].all(axis=1) & _pair_ok)[0]

        self.CP_Type = _cp_type[_keep_cp]
        self.CP_Pos = _cp_pos[_keep_cp] % self._box_size

        self.Node_CPIdx = np.where(self.CP_Type == 3)[0]
        self.Node_Pos = self.CP_Pos[self.Node_CPIdx]

        self.Filament_CPs = _new_id[_fil_cp[_keep_fil]]
        self.Filament_Arc = [_fil_pts[k] % self._box_size for k in _keep_fil]

        self.Filament_Length = np.array(
            [periodic_length(arc, self._box_size) for arc in self.Filament_Arc]
        )

    def _assign_nodes(self):
        _types = self.CP_Type
        _ncp = len(_types)
        _c1, _c2 = self.Filament_CPs.T
        _upper = _types != 2

        _e = _upper[_c1] & _upper[_c2]
        _G = coo_matrix((np.ones(_e.sum()), (_c1[_e], _c2[_e])), shape=(_ncp, _ncp))
        _, _comp = connected_components(_G, directed=False)

        _max_ids = np.nonzero(_types == 3)[0]
        _comp_max = np.full(_comp.max() + 1, -1)
        _comp_max[_comp[_max_ids]] = _max_ids

        _cp_node = _comp_max[_comp]
        _cp_node[_max_ids] = _max_ids
        _cp_node[~_upper] = -1

        _up = np.where((_types[_c1] == 3) | (_types[_c2] == 2), _c1, _c2)

        _cp_to_node = np.full(_ncp, -1)
        _cp_to_node[self.Node_CPIdx] = np.arange(len(self.Node_CPIdx))

        _n = _cp_node[_up]

        self.Filament_NodeIdx = np.where(_n >= 0, _cp_to_node[_n], -1)

    def _match_node_to_cluster(
        self, node_pos, cluster_pos, cluster_radius, n_candidates=32, n_max=2048
    ):
        _all_node_idx = np.arange(node_pos.shape[0])

        _node_cluster_idx, _node_cluster_norm_dist = periodic_cluster_match(
            node_pos,
            cluster_pos,
            cluster_radius,
            box_size=self._box_size,
            n_candidates=n_candidates,
            n_max=n_max,
        )

        return _all_node_idx[_node_cluster_norm_dist < 1.0], _node_cluster_idx

    def _get_segment_arrays(self, filament_arc):
        _seg_starts, _seg_ends, _seg_arc_idx = [], [], []
        for i, arc in enumerate(filament_arc):
            _seg_starts.append(arc[:-1])
            _seg_ends.append(arc[1:])
            _seg_arc_idx.append(np.full(arc.shape[0] - 1, i))

        return (
            np.concatenate(_seg_starts, axis=0),
            np.concatenate(_seg_ends, axis=0),
            np.concatenate(_seg_arc_idx, axis=0),
        )

    def _get_filament_eff_length(
        self, filament_arc, cluster_pos, cluster_radius, n_candidates=32, n_max=2048
    ):
        _seg_starts, _seg_ends, _seg_arc_idx = self._get_segment_arrays(filament_arc)

        _seg_mids = periodic_mean(
            np.stack([_seg_starts, _seg_ends], axis=1), box_size=self._box_size, axis=1
        )
        _seg_lengths = periodic_distance(_seg_starts, _seg_ends, self._box_size)

        _, _seg_cluster_norm_dist = periodic_cluster_match(
            _seg_mids,
            cluster_pos,
            cluster_radius,
            box_size=self._box_size,
            n_candidates=n_candidates,
            n_max=n_max,
        )

        _seg_out_cluster_mask = _seg_cluster_norm_dist > 1.0

        return np.bincount(
            _seg_arc_idx[_seg_out_cluster_mask],
            weights=_seg_lengths[_seg_out_cluster_mask],
            minlength=len(filament_arc),
        )

    def _get_galaxy_filament_distance(
        self, filament_arc, galaxy_pos, n_interp=3, n_candidates=16
    ):
        _seg_starts, _seg_ends, _seg_arc_idx = self._get_segment_arrays(filament_arc)

        _interp_points = periodic_interpolate(
            _seg_starts, _seg_ends, self._box_size, n_interp=n_interp
        )
        _interp_point_seg_indices = np.tile(
            np.arange(_seg_starts.shape[0]), n_interp + 2
        )

        _filament_tree = cKDTree(_interp_points, boxsize=self._box_size)
        _, _candidate_interp_point_indices = _filament_tree.query(
            galaxy_pos, k=n_candidates * (n_interp + 2)
        )

        _candidate_seg_indices = _interp_point_seg_indices[
            _candidate_interp_point_indices
        ]

        _candidate_seg_starts = _seg_starts[_candidate_seg_indices]
        _candidate_seg_ends = _seg_ends[_candidate_seg_indices]

        _galaxy_seg_distances = periodic_point_line_distance(
            galaxy_pos.reshape(-1, 1, 3),
            _candidate_seg_starts,
            _candidate_seg_ends,
            self._box_size,
        )

        _best_candidate_idx = np.argmin(_galaxy_seg_distances, axis=-1)
        _galalaxy_idx = np.arange(galaxy_pos.shape[0])

        galaxy_filament_dist = _galaxy_seg_distances[_galalaxy_idx, _best_candidate_idx]
        galaxy_filament_idx = _seg_arc_idx[
            _candidate_seg_indices[_galalaxy_idx, _best_candidate_idx]
        ]

        return galaxy_filament_dist, galaxy_filament_idx

    def _get_filament_effective_mass(
        self,
        N_filaments,
        galaxy_mass,
        galaxy_cluster_norm_dist,
        galaxy_filament_dist,
        galaxy_filament_idx,
        r_filament,
    ):
        _galaxy_mask = np.logical_and(
            galaxy_filament_dist < r_filament, galaxy_cluster_norm_dist > 1.0
        )

        _galaxy_mass = galaxy_mass[_galaxy_mask]
        _galaxy_filament_idx = galaxy_filament_idx[_galaxy_mask]

        return np.bincount(
            _galaxy_filament_idx,
            weights=_galaxy_mass,
            minlength=N_filaments,
        )

    def save_filament_arcs(self, filament_arcs_file):
        with open(filament_arcs_file, "w") as f:  # noqa: PTH123
            json.dump(
                {
                    "CP_Type": self.CP_Type.tolist(),
                    "CP_Pos": self.CP_Pos.tolist(),
                    "Node_CPIdx": self.Node_CPIdx.tolist(),
                    "Node_Pos": self.Node_Pos.tolist(),
                    "ClusterNode_Idx": self.ClusterNode_Idx.tolist(),
                    "Node_ClusterIdx": self.Node_ClusterIdx.tolist(),
                    "Filament_CPs": self.Filament_CPs.tolist(),
                    "Filament_Arc": [arc.tolist() for arc in self.Filament_Arc],
                    "Filament_NodeIdx": self.Filament_NodeIdx.tolist(),
                    "ClusterFilament_Idx": self.ClusterFilament_Idx.tolist(),
                    "Filament_ClusterIdx": self.Filament_ClusterIdx.tolist(),
                    "Filament_Length": self.Filament_Length.tolist(),
                    "Filament_EffLength": self.Filament_EffLength.tolist(),
                    "Filament_EffMass": self.Filament_EffMass.tolist(),
                    "Filament_EffMassPerLength": self.Filament_EffMassPerLength.tolist(),
                },
                f,
            )


class WallReader:
    def __init__(self, wall_file, box_size, galaxy_dict=None):
        self._box_size = box_size

        self.wall_triangles = self._read_wall_triangles(wall_file)

        if galaxy_dict is not None:
            self.Galaxy_WallDistance = self._get_galaxy_wall_distances(
                galaxy_dict["Position"]
            )

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

    def _get_galaxy_wall_distances(
        self, galaxy_pos, dense_midpoints=False, n_candidates=16
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
        _, _candidate_midpoint_indices = _wall_tree.query(
            galaxy_pos, k=n_candidates * (16 if dense_midpoints else 7)
        )

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
