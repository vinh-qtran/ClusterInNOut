import os

import h5py
import numpy as np
from tqdm import tqdm


class BaseReader:
    def __init__(self, group_dir, fof_file_base):
        _fof_files = self._get_fof_files(group_dir, fof_file_base)
        self._h = self._get_HubbleParams(_fof_files[0])

        for _file in tqdm(_fof_files, total=len(_fof_files)):
            _data = self._read_single_fof_file(_file)

            _keys = list(_data.keys())
            self._append_data(_data, _keys)

        self._keys = _keys

        for key in _keys:
            setattr(self, key, np.concatenate(getattr(self, key), axis=0))

    def _get_fof_files(self, group_dir, fof_file_base):
        fof_files = []

        for file in os.scandir(group_dir):
            if (
                file.is_file()
                and file.name.startswith(fof_file_base)
                and file.name.endswith(".hdf5")
            ):
                fof_files.append(file.path)  # noqa: PERF401

        return sorted(fof_files, key=lambda f: int(f.split("/")[-1].split(".")[1]))

    def _get_HubbleParams(self, base_file):
        with h5py.File(base_file, "r") as f:
            return f["Header"].attrs["HubbleParam"]

    def _read_single_fof_file(self, fof_file):
        _msg = "Not implemented in base class."
        raise NotImplementedError(_msg)

    def _append_data(self, data_dict, keys):
        for key in keys:
            if not hasattr(self, key):
                setattr(self, key, [])
            getattr(self, key).append(data_dict[key])

    def save_to_hdf5(self, output_file):
        with h5py.File(output_file, "w") as f:
            for key in self._keys:
                f.create_dataset(key, data=getattr(self, key))


class GroupReader(BaseReader):
    def __init__(self, M200_min=1e3, *args, **kwargs):
        self._M200_min = M200_min

        super().__init__(*args, **kwargs)

    def _read_single_fof_file(self, fof_file):
        with h5py.File(fof_file, "r") as f:
            if "GroupCM" not in f["Group"]:
                return {
                    "CM": np.empty((0, 3)),
                    "M200": np.empty(0),
                    "R200": np.empty(0),
                }

            CM = f["Group"]["GroupCM"][:] / self._h
            M200 = f["Group"]["Group_M_Crit200"][:] / self._h
            R200 = f["Group"]["Group_R_Crit200"][:] / self._h

            _mask = self._M200_min <= M200

        return {
            "CM": CM[_mask],
            "M200": M200[_mask],
            "R200": R200[_mask],
        }


class SubhaloReader(BaseReader):
    def __init__(self, Mdm_min=1e0, Mstar_min=1e-1, *args, **kwargs):
        self._Mdm_min = Mdm_min
        self._Mstar_min = Mstar_min

        super().__init__(*args, **kwargs)

    def _read_single_fof_file(self, fof_file):
        with h5py.File(fof_file, "r") as f:
            if "SubhaloCM" not in f["Subhalo"]:
                return {
                    "CM": np.empty((0, 3)),
                    "DMMass": np.empty(0),
                    "StellarMassRatio": np.empty(0),
                    "GasFraction": np.empty(0),
                    "sSFR": np.empty(0),
                    "Color": np.empty(0),
                    "GasMetallicity": np.empty(0),
                    "StellarMetallicity": np.empty(0),
                    "GasHalfMassRadius": np.empty(0),
                    "DMHalfMassRadius": np.empty(0),
                    "StellarHalfMassRadius": np.empty(0),
                    "SpinMagnitude": np.empty(0),
                    "VelocityDispersion": np.empty(0),
                    "Vmax": np.empty(0),
                    "VmaxRadius": np.empty(0),
                }

            # CM
            CM = f["Subhalo"]["SubhaloCM"][:] / self._h

            # DMMass
            DMMass = f["Subhalo"]["SubhaloMassType"][:, 1] / self._h

            # StellarMassRatio
            StellarMassRatio = (
                f["Subhalo"]["SubhaloMassType"][:, 4] / f["Subhalo"]["SubhaloMass"][:]
            )

            # GasFraction
            GasFraction = f["Subhalo"]["SubhaloMassType"][:, 0] / (
                f["Subhalo"]["SubhaloMass"][:] - f["Subhalo"]["SubhaloMassType"][:, 1]
            )

            # sSFR
            sSFR = f["Subhalo"]["SubhaloSFR"][:] / f["Subhalo"]["SubhaloMassType"][:, 4]

            # Color
            Color = (
                f["Subhalo"]["SubhaloStellarPhotometrics"][:, 4]
                - f["Subhalo"]["SubhaloStellarPhotometrics"][:, 5]
            )

            # GasMetallicity
            GasMetallicity = f["Subhalo"]["SubhaloGasMetallicity"][:]

            # StellarMetallicity
            StellarMetallicity = f["Subhalo"]["SubhaloStarMetallicity"][:]

            # GasHalfMassRadius
            GasHalfMassRadius = f["Subhalo"]["SubhaloHalfmassRadType"][:, 0] / self._h

            # DMHalfMassRadius
            DMHalfMassRadius = f["Subhalo"]["SubhaloHalfmassRadType"][:, 1] / self._h

            # StellarHalfMassRadius
            StellarHalfMassRadius = (
                f["Subhalo"]["SubhaloHalfmassRadType"][:, 4] / self._h
            )

            # SpinMagnitude
            SpinMagnitude = np.linalg.norm(f["Subhalo"]["SubhaloSpin"][:], axis=1)

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
                "CM": CM[_mask],
                "DMMass": DMMass[_mask],
                "StellarMassRatio": StellarMassRatio[_mask],
                "GasFraction": GasFraction[_mask],
                "sSFR": sSFR[_mask],
                "Color": Color[_mask],
                "GasMetallicity": GasMetallicity[_mask],
                "StellarMetallicity": StellarMetallicity[_mask],
                "GasHalfMassRadius": GasHalfMassRadius[_mask],
                "DMHalfMassRadius": DMHalfMassRadius[_mask],
                "StellarHalfMassRadius": StellarHalfMassRadius[_mask],
                "SpinMagnitude": SpinMagnitude[_mask],
                "VelocityDispersion": VelocityDispersion[_mask],
                "Vmax": Vmax[_mask],
                "VmaxRadius": VmaxRadius[_mask],
            }
