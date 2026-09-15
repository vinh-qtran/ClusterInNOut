import json
import os

import lightgbm as lgb
import numpy as np
from sklearn.metrics import confusion_matrix


class GBDTTraining:
    def __init__(
        self,
        num_classes: int,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 20,
        class_weights=None,
        params=None,
    ):

        self._num_classes = num_classes
        self._num_boost_round = num_boost_round
        self._early_stopping_rounds = early_stopping_rounds

        self._params = {
            "objective": "multiclass",
            "num_class": num_classes,
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbose": -1,
        }
        if params:
            self._params.update(params)

        if class_weights is not None:
            self._sample_weight_map = np.asarray(class_weights, dtype=np.float64)
        else:
            self._sample_weight_map = None

        self._model = None
        self._feature_names = None

    def _get_sample_weights(self, labels):
        if self._sample_weight_map is None:
            return None
        return self._sample_weight_map[labels]

    def train(
        self,
        train_features,
        train_labels,
        val_features,
        val_labels,
        out_path,
        feature_names=None,
    ):
        os.makedirs(out_path, exist_ok=True)  # noqa: PTH103

        self._feature_names = feature_names or [
            f"feature_{i}" for i in range(train_features.shape[1])
        ]

        _train_set = lgb.Dataset(
            train_features,
            label=train_labels,
            weight=self._get_sample_weights(train_labels),
            feature_name=self._feature_names,
        )
        _val_set = lgb.Dataset(
            val_features,
            label=val_labels,
            weight=self._get_sample_weights(val_labels),
            reference=_train_set,
        )

        _eval_results = {}

        self._model = lgb.train(
            self._params,
            _train_set,
            num_boost_round=self._num_boost_round,
            valid_sets=[_train_set, _val_set],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(self._early_stopping_rounds),
                lgb.log_evaluation(period=10),
                lgb.record_evaluation(_eval_results),
            ],
        )

        self._model.save_model(f"{out_path}/best_model.txt")

        with open(f"{out_path}/training_stats.json", "w") as f:  # noqa: PTH123
            json.dump(
                {
                    "train_multi_logloss": _eval_results["train"]["multi_logloss"],
                    "val_multi_logloss": _eval_results["val"]["multi_logloss"],
                    "best_iteration": self._model.best_iteration,
                },
                f,
            )

    def test(self, test_features, test_labels, model_path=None, out_path=None):
        if model_path is not None:
            self._model = lgb.Booster(model_file=model_path)

        _probs = self._model.predict(
            test_features, num_iteration=self._model.best_iteration
        )
        _preds = np.argmax(_probs, axis=1)

        _confusion_matrix = confusion_matrix(
            test_labels, _preds, labels=range(self._num_classes), normalize="true"
        )

        if out_path is not None:
            with open(f"{out_path}/test_results.json", "w") as f:  # noqa: PTH123
                json.dump(_confusion_matrix.tolist(), f)

    def feature_importance(self, importance_type="gain", out_path=None):
        _importances = self._model.feature_importance(importance_type=importance_type)
        _pairs = sorted(
            zip(self._feature_names, _importances, strict=False),
            key=lambda x: x[1],
            reverse=True,
        )

        if out_path is not None:
            with open(f"{out_path}/feature_importance.json", "w") as f:  # noqa: PTH123
                json.dump(dict(_pairs), f)
