import numpy as np


class EnvironmentMask:
    def __init__(
        self,
        cluster_norm_distance,
        filament_norm_distance,
        wall_distance,
        tidal_ratio,
        cluster_thresholds=(0.5, 1.0, 1.5),
        filament_thresholds=(0.5, 2.0),
        wall_thresholds=(1e3, 2e3),
        tidal_threshold=3.0,
    ):

        self._cluster_norm_distance = cluster_norm_distance
        self._filament_norm_distance = filament_norm_distance
        self._wall_distance = wall_distance

        self._tidal_ratio = tidal_ratio

        self._cluster_thresholds = cluster_thresholds
        self._filament_thresholds = filament_thresholds
        self._wall_thresholds = wall_thresholds

        self._tidal_threshold = tidal_threshold

    def _get_isotropic_cluster_filament_mask(self):
        cluster_mask = self._cluster_norm_distance < self._cluster_thresholds[0]

        cluster_outskirt_mask = np.logical_and(
            self._cluster_norm_distance > self._cluster_thresholds[0],
            self._cluster_norm_distance < self._cluster_thresholds[-1],
        )

        filament_mask = np.logical_and(
            self._cluster_norm_distance > self._cluster_thresholds[-1],
            self._filament_norm_distance < self._filament_thresholds[0],
        )

        filament_outskirt_mask = np.logical_and(
            self._cluster_norm_distance > self._cluster_thresholds[-1],
            np.logical_and(
                self._filament_norm_distance > self._filament_thresholds[0],
                self._filament_norm_distance < self._filament_thresholds[-1],
            ),
        )

        return (
            cluster_mask,
            cluster_outskirt_mask,
            filament_mask,
            filament_outskirt_mask,
        )

    def _get_tidal_cluster_filament_mask(self):
        cluster_mask = self._cluster_norm_distance < self._cluster_thresholds[0]

        _shared_outskirt_mask = np.logical_and(
            self._cluster_norm_distance > self._cluster_thresholds[1],
            self._cluster_norm_distance < self._cluster_thresholds[-1],
        )

        cluster_outskirt_mask = np.logical_or(
            np.logical_and(
                self._cluster_norm_distance > self._cluster_thresholds[0],
                self._cluster_norm_distance < self._cluster_thresholds[1],
            ),
            np.logical_and(
                _shared_outskirt_mask, self._tidal_ratio > self._tidal_threshold
            ),
        )

        filament_mask = np.logical_or(
            np.logical_and(
                self._cluster_norm_distance > self._cluster_thresholds[-1],
                self._filament_norm_distance < self._filament_thresholds[0],
            ),
            np.logical_and(
                np.logical_and(
                    _shared_outskirt_mask, self._tidal_ratio < 1 / self._tidal_threshold
                ),
                self._filament_norm_distance < self._filament_thresholds[0],
            ),
        )

        filament_outskirt_mask = np.logical_or(
            np.logical_and(
                self._cluster_norm_distance > self._cluster_thresholds[-1],
                np.logical_and(
                    self._filament_norm_distance > self._filament_thresholds[0],
                    self._filament_norm_distance < self._filament_thresholds[-1],
                ),
            ),
            np.logical_and(
                np.logical_and(
                    _shared_outskirt_mask, self._tidal_ratio < 1 / self._tidal_threshold
                ),
                np.logical_and(
                    self._filament_norm_distance > self._filament_thresholds[0],
                    self._filament_norm_distance < self._filament_thresholds[-1],
                ),
            ),
        )

        return (
            cluster_mask,
            cluster_outskirt_mask,
            filament_mask,
            filament_outskirt_mask,
        )

    def _get_wall_mask(self):
        return np.logical_and(
            np.logical_and(
                self._cluster_norm_distance > self._cluster_thresholds[-1],
                self._filament_norm_distance > self._filament_thresholds[-1],
            ),
            self._wall_distance < self._wall_thresholds[0],
        )

    def _get_void_mask(self):
        return np.logical_and(
            np.logical_and(
                self._cluster_norm_distance > self._cluster_thresholds[-1],
                self._filament_norm_distance > self._filament_thresholds[-1],
            ),
            self._wall_distance > self._wall_thresholds[-1],
        )

    def _get_wall_void_mask(self):
        return np.logical_and(
            self._cluster_norm_distance > self._cluster_thresholds[-1],
            self._filament_norm_distance > self._filament_thresholds[-1],
        )

    def get_isotropic_masks(self):
        cluster_mask, cluster_outskirt_mask, filament_mask, filament_outskirt_mask = (
            self._get_isotropic_cluster_filament_mask()
        )
        wall_mask = self._get_wall_mask()
        void_mask = self._get_void_mask()
        wall_void_mask = self._get_wall_void_mask()

        return {
            "cluster": cluster_mask,
            "cluster_outskirt": cluster_outskirt_mask,
            "filament": filament_mask,
            "filament_outskirt": filament_outskirt_mask,
            "wall": wall_mask,
            "void": void_mask,
            "wall_void": wall_void_mask,
        }

    def get_tidal_masks(self):
        cluster_mask, cluster_outskirt_mask, filament_mask, filament_outskirt_mask = (
            self._get_tidal_cluster_filament_mask()
        )
        wall_mask = self._get_wall_mask()
        void_mask = self._get_void_mask()
        wall_void_mask = self._get_wall_void_mask()

        return {
            "cluster": cluster_mask,
            "cluster_outskirt": cluster_outskirt_mask,
            "filament": filament_mask,
            "filament_outskirt": filament_outskirt_mask,
            "wall": wall_mask,
            "void": void_mask,
            "wall_void": wall_void_mask,
        }
