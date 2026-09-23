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
        for _param in [
            "StellarMassRatio",
            "VelocityDispersion",
            "Vmax",
            "VmaxRadius",
            "Velocity",
            "Spin",
            "DMMass",
            "DMHalfMassRadius",
            "StellarHalfMassRadius",
        ]:
            setattr(self, _param, np.log10(getattr(self, _param)))

        self.sSFR = np.arcsinh(self.sSFR - 1)

        self.GasHalfMassRadius = np.arcsinh(self.GasHalfMassRadius / 50 - 1)

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
        pass

    def get_train_val_test_masks(self, train_frac=0.7, val_frac=0.15, box_size=302627):
        _train_mask = np.logical_or(
            self.Position[:, 0] < (train_frac / 2) * box_size,
            self.Position[:, 0] > (1 - train_frac / 2) * box_size,
        )

        _val_mask = np.logical_or(
            np.logical_and(
                self.Position[:, 0] > (train_frac / 2) * box_size + 2e3,
                self.Position[:, 0] < (train_frac / 2 + val_frac / 2) * box_size + 2e3,
            ),
            np.logical_and(
                self.Position[:, 0] < (1 - train_frac / 2) * box_size - 2e3,
                self.Position[:, 0]
                > (1 - train_frac / 2 - val_frac / 2) * box_size - 2e3,
            ),
        )

        _test_mask = np.logical_and(
            self.Position[:, 0] > (train_frac / 2 + val_frac / 2) * box_size + 4e3,
            self.Position[:, 0] < (1 - train_frac / 2 - val_frac / 2) * box_size - 4e3,
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

    def get_loaders(self):
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
