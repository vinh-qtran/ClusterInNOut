import os

import h5py
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


class BaseReader:
    def __init__(self, group_dir, fof_file_base):
        _fof_files = self._get_fof_files(group_dir, fof_file_base)
        self._h, self._box_size, self._a, self._t_H = self._read_header(_fof_files[0])

        for _file in tqdm(_fof_files, total=len(_fof_files)):
            _data = self._read_single_fof_file(_file)

            _keys = list(_data.keys())
            self._append_data(_data, _keys)

        self._keys = _keys

        for key in _keys:
            setattr(self, key, np.concatenate(getattr(self, key), axis=0))

    def _get_fof_files(self, group_dir, fof_file_base):
        fof_files = []

        with os.scandir(group_dir) as _it:
            for _file in _it:
                if (
                    _file.is_file()
                    and _file.name.startswith(fof_file_base)
                    and _file.name.endswith(".hdf5")
                ):
                    fof_files.append(_file.path)  # noqa: PERF401

        return sorted(fof_files, key=lambda f: int(f.split("/")[-1].split(".")[1]))

    def _read_header(self, base_file):
        with h5py.File(base_file, "r") as f:
            _header = f["Header"].attrs

            h = _header["HubbleParam"]
            box_size = _header["BoxSize"] / h

            a = _header["Time"]

            _Omega0 = _header["Omega0"]
            _OmegaLambda = _header["OmegaLambda"]

            _Ea = np.sqrt(_Omega0 * a**-3 + _OmegaLambda)

            t_H = 9.7779 / h / _Ea

        return h, box_size, a, t_H

    def _read_single_fof_file(self, fof_file):
        _msg = "Not implemented in base class."
        raise NotImplementedError(_msg)

    def _append_data(self, data_dict, keys):
        for key in keys:
            if not hasattr(self, key):
                setattr(self, key, [])
            getattr(self, key).append(data_dict[key])

    def save_to_hdf5(self, output_file, keys=None):
        keys = keys or self._keys
        with h5py.File(output_file, "w") as f:
            for key in keys:
                f.create_dataset(key, data=getattr(self, key))


class GroupReader(BaseReader):
    def __init__(self, M200c_min=1e3, *args, **kwargs):
        self._M200c_min = M200c_min

        super().__init__(*args, **kwargs)

    def _read_single_fof_file(self, fof_file):
        with h5py.File(fof_file, "r") as f:
            if "Group/GroupPos" not in f:
                return {
                    "Position": np.empty((0, 3)),
                    "M200c": np.empty(0),
                    # "R200c": np.empty(0),
                    "R200m": np.empty(0),
                }

            Position = f["Group"]["GroupPos"][:] / self._h
            M200c = f["Group"]["Group_M_Crit200"][:] / self._h
            # R200c = f["Group"]["Group_R_Crit200"][:] / self._h
            R200m = f["Group"]["Group_R_Mean200"][:] / self._h

            _mask = self._M200c_min <= M200c

        return {
            "Position": Position[_mask],
            "M200c": M200c[_mask],
            # "R200c": R200c[_mask],
            "R200m": R200m[_mask],
        }


class SubhaloReader(BaseReader):
    def __init__(self, Mdm_min=5e-1, Mstar_min=1e-1, *args, **kwargs):
        self._Mdm_min = Mdm_min
        self._Mstar_min = Mstar_min

        super().__init__(*args, **kwargs)

    def _read_single_fof_file(self, fof_file):
        with h5py.File(fof_file, "r") as f:
            if "Subhalo/SubhaloPos" not in f:
                return {
                    "Position": np.empty((0, 3)),
                    "Velocity": np.empty((0, 3)),
                    "Spin": np.empty((0, 3)),
                    "DMMass": np.empty(0),
                    "StellarMassRatio": np.empty(0),
                    "GasFraction": np.empty(0),
                    "sSFR": np.empty(0),
                    "Color": np.empty(0),
                    "StellarMetallicity": np.empty(0),
                    "GasMetallicity": np.empty(0),
                    "DMHalfMassRadius": np.empty(0),
                    "StellarHalfMassRadius": np.empty(0),
                    "GasHalfMassRadius": np.empty(0),
                    "VelocityDispersion": np.empty(0),
                    "Vmax": np.empty(0),
                    "VmaxRadius": np.empty(0),
                }

            # Position
            Position = f["Subhalo"]["SubhaloPos"][:] / self._h

            # Velocity
            Velocity = f["Subhalo"]["SubhaloVel"][:]

            # Spin
            Spin = f["Subhalo"]["SubhaloSpin"][:] / self._h

            # DMMass
            DMMass = f["Subhalo"]["SubhaloMassType"][:, 1] / self._h

            # StellarMassRatio
            StellarMassRatio = (
                f["Subhalo"]["SubhaloMassType"][:, 4] / f["Subhalo"]["SubhaloMass"][:]
            )

            # GasFraction
            GasFraction = f["Subhalo"]["SubhaloMassType"][:, 0] / (
                f["Subhalo"]["SubhaloMassType"][:, 0]
                + f["Subhalo"]["SubhaloMassType"][
                    :, 4
                ]  # + f["Subhalo"]["SubhaloMassType"][:, 5]
            )

            # sSFR
            sSFR = (
                f["Subhalo"]["SubhaloSFR"][:]
                / f["Subhalo"]["SubhaloMassType"][:, 4]
                * self._h
                / 1e1
                * self._t_H
            )

            # Color
            Color = (
                f["Subhalo"]["SubhaloStellarPhotometrics"][:, 4]
                - f["Subhalo"]["SubhaloStellarPhotometrics"][:, 5]
            )

            # StellarMetallicity
            StellarMetallicity = f["Subhalo"]["SubhaloStarMetallicity"][:] / 0.0127

            # GasMetallicity
            GasMetallicity = f["Subhalo"]["SubhaloGasMetallicity"][:] / 0.0127

            # DMHalfMassRadius
            DMHalfMassRadius = f["Subhalo"]["SubhaloHalfmassRadType"][:, 1] / self._h

            # StellarHalfMassRadius
            StellarHalfMassRadius = (
                f["Subhalo"]["SubhaloHalfmassRadType"][:, 4] / self._h
            )

            # GasHalfMassRadius
            GasHalfMassRadius = f["Subhalo"]["SubhaloHalfmassRadType"][:, 0] / self._h

            # VelocityDispersion
            VelocityDispersion = f["Subhalo"]["SubhaloVelDisp"][:]

            # Vmax
            Vmax = f["Subhalo"]["SubhaloVmax"][:]

            # VmaxRadius
            VmaxRadius = f["Subhalo"]["SubhaloVmaxRad"][:] / self._h

            _mask = np.logical_and(
                np.logical_and(
                    f["Subhalo"]["SubhaloMassType"][:, 1] / self._h >= self._Mdm_min,
                    f["Subhalo"]["SubhaloMassType"][:, 4] / self._h >= self._Mstar_min,
                ),
                f["Subhalo"]["SubhaloFlag"][:] == 1,
            )

            return {
                "Position": Position[_mask],
                "Velocity": Velocity[_mask],
                "Spin": Spin[_mask],
                "DMMass": DMMass[_mask],
                "StellarMassRatio": StellarMassRatio[_mask],
                "GasFraction": GasFraction[_mask],
                "sSFR": sSFR[_mask],
                "Color": Color[_mask],
                "StellarMetallicity": StellarMetallicity[_mask],
                "GasMetallicity": GasMetallicity[_mask],
                "DMHalfMassRadius": DMHalfMassRadius[_mask],
                "StellarHalfMassRadius": StellarHalfMassRadius[_mask],
                "GasHalfMassRadius": GasHalfMassRadius[_mask],
                "VelocityDispersion": VelocityDispersion[_mask],
                "Vmax": Vmax[_mask],
                "VmaxRadius": VmaxRadius[_mask],
            }

    def match_to_groups(self, group_pos):
        _tree = cKDTree(group_pos, boxsize=self._box_size)

        group_dist, group_idx = _tree.query(self.Position)

        self.ClusterIdx = group_idx
        self.ClusterDistance = group_dist

    def get_filament_distances(self, filament_arcs, n_interp=100):
        _arc_midpoints = np.concatenate(
            [
                np.concatenate(
                    [
                        ((n_interp - i) * arc[:-1] + i * arc[1:]) / n_interp
                        for i in range(1, n_interp + 1)
                    ],
                    axis=0,
                )
                for arc in filament_arcs
            ],
            axis=0,
        )

        _tree = cKDTree(_arc_midpoints, boxsize=self._box_size)

        filament_dist, _ = _tree.query(self.Position)

        self.FilamentDistance = filament_dist

    def get_wall_distances(self, wall_triangles):
        _triangle_midpoints = np.concatenate(
            [
                wall_triangles.reshape(-1, 3),
                wall_triangles.mean(axis=1),
                wall_triangles[:, [0, 1]].mean(axis=1),
                wall_triangles[:, [1, 2]].mean(axis=1),
                wall_triangles[:, [2, 0]].mean(axis=1),
                wall_triangles[:, [0, 0, 0, 1]].mean(axis=1),
                wall_triangles[:, [1, 1, 1, 2]].mean(axis=1),
                wall_triangles[:, [2, 2, 2, 0]].mean(axis=1),
                wall_triangles[:, [0, 1, 1, 1]].mean(axis=1),
                wall_triangles[:, [1, 2, 2, 2]].mean(axis=1),
                wall_triangles[:, [2, 0, 0, 0]].mean(axis=1),
                wall_triangles[:, [0, 1, 2, 0]].mean(axis=1),
                wall_triangles[:, [1, 2, 0, 1]].mean(axis=1),
                wall_triangles[:, [2, 0, 1, 2]].mean(axis=1),
            ],
            axis=0,
        )

        # _triangle_midpoints = _triangle_midpoints % self._box_size

        _tree = cKDTree(_triangle_midpoints, boxsize=self._box_size)

        wall_dist, _ = _tree.query(self.Position)

        self.WallDistance = wall_dist
