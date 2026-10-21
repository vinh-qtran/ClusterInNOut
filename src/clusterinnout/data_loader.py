import h5py
import numpy as np

from clusterinnout.label_utils import EnvironmentMask


class GalaxyData:
    def __init__(self, galaxy_file, **label_kwargs):
        self._get_galaxy_data(galaxy_file)

        self._env_mask = EnvironmentMask(
            cluster_norm_distance=self.ClusterNormDistance,
            filament_norm_distance=self.FilamentNormDistance,
            wall_distance=self.WallDistance,
            tidal_ratio=self.ClusterTidal / self.FilamentTidal,
            **label_kwargs,
        )

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
        return self._env_mask.get_isotropic_masks()

    def get_tidal_masks(self):
        return self._env_mask.get_tidal_masks()

    def get_train_val_test_masks(
        self,
        fold=0,
        n_folds=5,
        test_frac=0.15,
        test_center=None,
        buffer=5e3,
        box_size=302627,
    ):
        """Spatial k-fold split along x in the periodic box.

        The test slab is fixed for every fold: the test_frac * box_size wide slab of x
        centred on test_center (default: box centre). The rest of the box is one
        contiguous arc that wraps through x = 0. A gap of `buffer` is dropped on each side
        of the test slab, and what is left of the arc is cut into n_folds equal chunks:
        fold k uses chunk k as val and the other chunks as train, with a gap of `buffer`
        between train and val.

        Across the n_folds folds, every galaxy outside the test slab and its gaps is in
        val exactly once. The first and last folds have only one train-val gap (their
        val chunk borders a test gap), so they train on ~1 buffer width more data.
        """
        if not 0 <= fold < n_folds:
            _msg = f"fold must be in [0, {n_folds - 1}], got {fold}"
            raise ValueError(_msg)
        if test_center is None:
            test_center = box_size / 2

        _test_width = test_frac * box_size
        _test_start = test_center - _test_width / 2

        # distance along x from the start of the test slab, wrapped into [0, box_size)
        _u = (self.Position[:, 0] - _test_start) % box_size
        _test_mask = _u < _test_width

        # position along the train/val arc: 0 at the end of the test slab
        # (test galaxies get negative values and so fall outside every arc range below)
        _arc = _u - _test_width
        _arc_length = box_size - _test_width
        _edges = np.linspace(buffer, _arc_length - buffer, n_folds + 1)
        _val_start, _val_end = _edges[fold], _edges[fold + 1]

        _usable = (_arc >= _edges[0]) & (_arc < _edges[-1])
        _val_mask = (_arc >= _val_start) & (_arc < _val_end)
        _train_mask = _usable & (
            (_arc < _val_start - buffer) | (_arc >= _val_end + buffer)
        )

        return {
            "train": _train_mask,
            "val": _val_mask,
            "test": _test_mask,
        }
