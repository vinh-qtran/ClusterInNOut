import os

import h5py
import numpy as np
from tqdm import tqdm

from clusterinnout.distance_utils import periodic_cluster_match
from clusterinnout.splash_back_utils import Diemer20_Rsp_R200m_scaler


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
    def __init__(self, M200c_min=1e3, Rsp_R200m_scaler=1, **kwargs):
        self._M200c_min = M200c_min

        self._Rsp_R200m_scaler = Rsp_R200m_scaler

        super().__init__(**kwargs)

    def _read_single_fof_file(self, fof_file):
        with h5py.File(fof_file, "r") as f:
            if "Group/GroupPos" not in f:
                return {
                    "Position": np.empty((0, 3)),
                    "M200c": np.empty(0),
                    "R200c": np.empty(0),
                    "M200m": np.empty(0),
                    "R200m": np.empty(0),
                    "Rsp": np.empty(0),
                }

            Position = f["Group"]["GroupPos"][:] / self._h

            M200c = f["Group"]["Group_M_Crit200"][:] / self._h
            R200c = f["Group"]["Group_R_Crit200"][:] / self._h

            M200m = f["Group"]["Group_M_Mean200"][:] / self._h
            R200m = f["Group"]["Group_R_Mean200"][:] / self._h

            _mask = M200c > self._M200c_min

            if _mask.sum() == 0:
                _Rsp_R200m_scaler = np.zeros_like(R200m)
            elif isinstance(self._Rsp_R200m_scaler, (int, float)):
                _Rsp_R200m_scaler = self._Rsp_R200m_scaler
            elif (
                isinstance(self._Rsp_R200m_scaler, str)
                and self._Rsp_R200m_scaler == "Diemer20"
            ):
                _Rsp_R200m_scaler = Diemer20_Rsp_R200m_scaler(
                    M200m * 1e10, 1 / self._a - 1
                )
            else:
                _msg = f"Invalid Rsp_R200m_scaler: {self._Rsp_R200m_scaler}"
                raise ValueError(_msg)

            Rsp = _Rsp_R200m_scaler * R200m

        return {
            "Position": Position[_mask],
            "M200c": M200c[_mask],
            "R200c": R200c[_mask],
            "M200m": M200m[_mask],
            "R200m": R200m[_mask],
            "Rsp": Rsp[_mask],
        }


class SubhaloReader(BaseReader):
    def __init__(self, M_min=None, Mdm_min=6e-1, Mstar_min=3e-2, **kwargs):
        self._M_min = M_min
        self._Mdm_min = Mdm_min
        self._Mstar_min = Mstar_min

        super().__init__(**kwargs)

    def _read_single_fof_file(self, fof_file):
        with h5py.File(fof_file, "r") as f:
            if "Subhalo/SubhaloPos" not in f:
                return {
                    "Position": np.empty((0, 3)),
                    "Mass": np.empty(0),
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

            # Mass
            Mass = f["Subhalo"]["SubhaloMass"][:] / self._h

            # Velocity
            Velocity = f["Subhalo"]["SubhaloVel"][:]

            # Spin
            Spin = f["Subhalo"]["SubhaloSpin"][:] / self._h

            # DMMass
            DMMass = f["Subhalo"]["SubhaloMassType"][:, 1] / self._h

            # StellarMassRatio
            StellarMassRatio = (
                f["Subhalo"]["SubhaloMassInRadType"][:, 4]
                / f["Subhalo"]["SubhaloMass"][:]
            )

            # GasFraction
            GasFraction = (f["Subhalo"]["SubhaloMassInRadType"][:, 0]) / (
                f["Subhalo"]["SubhaloMassInRadType"][:, 0]
                + f["Subhalo"]["SubhaloMassInRadType"][:, 4]
            )

            # sSFR
            sSFR = (
                f["Subhalo"]["SubhaloSFRinRad"][:]
                / f["Subhalo"]["SubhaloMassInRadType"][:, 4]
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

            if self._M_min is not None:
                _mask = Mass > self._M_min
            else:
                _mask = np.logical_and(
                    np.logical_and(
                        f["Subhalo"]["SubhaloMassType"][:, 1] / self._h
                        >= self._Mdm_min,
                        f["Subhalo"]["SubhaloMassInRadType"][:, 4] / self._h
                        >= self._Mstar_min,
                    ),
                    f["Subhalo"]["SubhaloFlag"][:] == 1,
                )

            return {
                "Position": Position[_mask],
                "Mass": Mass[_mask],
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

    def match_to_cluster(
        self, cluster_pos, cluster_radius, n_candidates=32, n_max=2048
    ):
        cluster_idx, cluster_norm_dist = periodic_cluster_match(
            self.Position,
            cluster_pos,
            cluster_radius,
            box_size=self._box_size,
            n_candidates=n_candidates,
            n_max=n_max,
        )

        self.ClusterIdx = cluster_idx
        self.ClusterNormDistance = cluster_norm_dist
        self.ClusterDistance = cluster_norm_dist * cluster_radius[cluster_idx]
