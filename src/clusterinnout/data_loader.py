import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class GalaxyData:
    def __init__(self, galaxy_file):
        self._get_galaxy_data(galaxy_file)

        self._transform_features()

    def _get_galaxy_data(self, galaxy_file):
        with h5py.File(galaxy_file, "r") as f:
            for _param in f:
                if _param not in ["Velocity", "Spin"]:
                    setattr(self, _param, f[_param][:])
                else:
                    setattr(self, _param, np.linalg.norm(f[_param][:], axis=1))

    def _transform_features(self):
        _quenched_mask = self.sSFR == 0

        self.Quenched = _quenched_mask.astype(int)
        self.sSFR[_quenched_mask] = 10 ** (-4.5)

        self.GasDepleted = (self.GasFraction == 0).astype(int)
        self.GasHalfMassRadius[self.GasHalfMassRadius == 0] = 10 ** (-0.25)

        for _param in [
            "StellarMassRatio",
            "sSFR",
            "VelocityDispersion",
            "Vmax",
            "VmaxRadius",
            "Velocity",
            "Spin",
            "DMMass",
            "DMHalfMassRadius",
            "StellarHalfMassRadius",
            "GasHalfMassRadius",
        ]:
            setattr(self, _param, np.log10(getattr(self, _param)))

    def get_isotropic_masks(self):
        _cluster_mask = self.ClusterNormDistance < 0.5

        _cluster_outskirts_mask = np.logical_and(
            self.ClusterNormDistance > 0.5,
            self.ClusterNormDistance < 2.0,
        )

        _filament_mask = np.logical_and(
            self.ClusterNormDistance > 2.0,
            self.FilamentNormDistance < 0.5,
        )

        _filament_outskirts_mask = np.logical_and(
            self.ClusterNormDistance > 2.0,
            np.logical_and(
                self.FilamentNormDistance > 0.5,
                self.FilamentNormDistance < 2.0,
            ),
        )

        _wall_mask = np.logical_and(
            np.logical_and(
                self.ClusterNormDistance > 2.0,
                self.FilamentNormDistance > 2.0,
            ),
            self.WallDistance < 1e3,
        )

        _void_mask = np.logical_and(
            np.logical_and(
                self.ClusterNormDistance > 2.0,
                self.FilamentNormDistance > 2.0,
            ),
            self.WallDistance > 3e3,
        )

        _wall_void_mask = np.logical_and(
            self.ClusterNormDistance > 2.0,
            self.FilamentNormDistance > 2.0,
        )

        return {
            "cluster": _cluster_mask,
            "cluster_outskirts": _cluster_outskirts_mask,
            "filament": _filament_mask,
            "filament_outskirts": _filament_outskirts_mask,
            "wall": _wall_mask,
            "void": _void_mask,
            "wall_void": _wall_void_mask,
        }

    def get_tidal_masks(self):
        _core_cluster_mask = self.ClusterNormDistance < 0.5
        _core_filament_mask = np.logical_and(
            self.FilamentNormDistance < 0.5,
            self.ClusterNormDistance > 0.5,
        )
        _mid_mask = np.logical_or(
            np.logical_and(
                self.ClusterNormDistance > 0.5, self.ClusterNormDistance < 1.0
            ),
            np.logical_and(
                np.logical_and(
                    self.FilamentNormDistance > 0.5, self.FilamentNormDistance < 1.0
                ),
                self.ClusterNormDistance > 0.5,
            ),
        )

        _tidal_ratio = self.ClusterTidal / self.FilamentTidal

        _cluster_mask = np.logical_or(
            _core_cluster_mask, np.logical_and(_mid_mask, _tidal_ratio > 1.0)
        )

        _filament_mask = np.logical_or(
            _core_filament_mask, np.logical_and(_mid_mask, _tidal_ratio < 1.0)
        )

        _wall_mask = np.logical_and(
            np.logical_and(
                self.ClusterNormDistance > 2.0,
                self.FilamentNormDistance > 2.0,
            ),
            self.WallDistance < 1e3,
        )

        _void_mask = np.logical_and(
            np.logical_and(
                self.ClusterNormDistance > 2.0,
                self.FilamentNormDistance > 2.0,
            ),
            self.WallDistance > 3e3,
        )

        return {
            "cluster": _cluster_mask,
            "filament": _filament_mask,
            "wall": _wall_mask,
            "void": _void_mask,
        }

    def get_train_val_test_masks(
        self, off_set=0, train_frac=0.7, val_frac=0.15, box_size=302627
    ):
        _position = (self.Position + off_set) % box_size

        _train_mask = np.logical_or(
            _position[:, 0] < (train_frac / 2) * box_size,
            _position[:, 0] > (1 - train_frac / 2) * box_size,
        )

        _val_mask = np.logical_or(
            np.logical_and(
                _position[:, 0] > (train_frac / 2) * box_size + 2e3,
                _position[:, 0] < (train_frac / 2 + val_frac / 2) * box_size + 2e3,
            ),
            np.logical_and(
                _position[:, 0] < (1 - train_frac / 2) * box_size - 2e3,
                _position[:, 0] > (1 - train_frac / 2 - val_frac / 2) * box_size - 2e3,
            ),
        )

        _test_mask = np.logical_and(
            _position[:, 0] > (train_frac / 2 + val_frac / 2) * box_size + 4e3,
            _position[:, 0] < (1 - train_frac / 2 - val_frac / 2) * box_size - 4e3,
        )

        return {
            "train": _train_mask,
            "val": _val_mask,
            "test": _test_mask,
        }


class DataLoaderBuilder:
    def __init__(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: np.ndarray,
        val_labels: np.ndarray,
        test_features: np.ndarray,
        test_labels: np.ndarray,
        batch_size: int = 640,
        num_workers: int = 8,
        seed: int = 42,
    ):
        torch.manual_seed(seed)

        self._train_dataset = TensorDataset(
            torch.as_tensor(train_features, dtype=torch.float32),
            torch.as_tensor(train_labels, dtype=torch.long),
        )
        self._val_dataset = TensorDataset(
            torch.as_tensor(val_features, dtype=torch.float32),
            torch.as_tensor(val_labels, dtype=torch.long),
        )
        self._test_dataset = TensorDataset(
            torch.as_tensor(test_features, dtype=torch.float32),
            torch.as_tensor(test_labels, dtype=torch.long),
        )

        self._batch_size = batch_size
        self._num_workers = num_workers

    def get_train_val_test_loaders(self):
        train_loader = DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )

        val_loader = DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )

        test_loader = DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )

        return train_loader, val_loader, test_loader

    def get_input_loader(self, features):
        _data_set = TensorDataset(
            torch.as_tensor(features, dtype=torch.float32),
        )

        return DataLoader(
            _data_set,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )
