import glob
import json
import os
import warnings

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# ------------------------------------------------------------------------------
# defaults: 3 classes x 25k-50k training rows each (~75k-150k rows), tabular features
# ------------------------------------------------------------------------------
DEFAULT_PARAMS = {
    # loss: softmax cross-entropy (one tree per class per round, summed logits -> softmax)
    "objective": "multiclass",
    "metric": ["multi_logloss", "multi_error"],  # the first one drives early stopping
    # boosting: small steps, the number of rounds is chosen by early stopping
    "learning_rate": 0.03,
    # tree size: 63 leaves leaves room for feature interactions at this data size;
    # min_data_in_leaf ~0.1% of the rows keeps leaves from fitting noise
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 100,
    "min_sum_hessian_in_leaf": 1e-2,
    "min_gain_to_split": 0.0,
    # regularisation: L2 shrinks the leaf logits, which also curbs over-confidence
    "lambda_l1": 0.0,
    "lambda_l2": 1.0,
    # randomness: row / column subsampling per tree (bagging needs bagging_freq > 0)
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    # histogram resolution per feature
    "max_bin": 255,
    # misc
    "seed": 42,
    "num_threads": 0,  # 0 = all cores
    "verbosity": -1,
}

# name: ("int" | "float", low, high, log_scale)  or  ("categorical", [choices])
DEFAULT_SEARCH_SPACE = {
    "num_leaves": ("int", 15, 255, True),
    "min_data_in_leaf": ("int", 20, 1000, True),
    "min_sum_hessian_in_leaf": ("float", 1e-3, 1.0, True),
    "feature_fraction": ("float", 0.4, 1.0, False),
    "bagging_fraction": ("float", 0.5, 1.0, False),
    "lambda_l1": ("float", 1e-3, 10.0, True),
    "lambda_l2": ("float", 1e-3, 30.0, True),
    "min_gain_to_split": ("float", 0.0, 1.0, False),
}


class GBTreeTraining:
    """LightGBM softmax cross-entropy classifier with equal-mix (class-weighted) metrics.

    Workflow
        gbt = GBTreeTraining(X_tr, y_tr, X_va, y_va, num_classes=3,
                             calib_features=X_cal, calib_labels=y_cal)
        gbt.tune("runs/tune")                   # optional; updates the params in place
        gbt.train("runs/single")                # or gbt.train_kfold("runs/kfold", folds)
        T = gbt.fit_temperature("runs/single", out_path="runs/single")
        labels, probs = gbt.test(X_te, y_te, "runs/single", temperature=T, out_path=...)
        shap = gbt.shap_values(X_te, "runs/single", labels=y_te, temperature=T, ...)

    model_path is a model file, or an out_path written by train() (best_model.txt) or
    by train_kfold() (fold_*/best_model.txt; the folds' logits are averaged). Use
    separate out_paths for the two.
    """

    def __init__(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: np.ndarray | None,
        val_labels: np.ndarray | None,
        num_classes: int = 3,
        calib_features: np.ndarray | None = None,
        calib_labels: np.ndarray | None = None,
        num_boost_round: int = 5000,
        early_stopping_rounds: int | None = 200,
        params=None,
        class_weights=None,
        feature_names=None,
        log_period: int = 100,
    ):
        """
        train_* / val_*  : (n, F) features and (n,) integer labels 0..K-1. val may be None
                           if you only use train_kfold / tune(folds=...).
        calib_*          : optional held-out set used ONLY to fit the temperature.
        num_boost_round  : upper bound on boosting rounds (one tree per class per round);
                           early stopping on the val equal-mix log-loss picks the best.
        params           : LightGBM params, merged over DEFAULT_PARAMS. The objective must
                           stay softmax cross-entropy ("multiclass").
        class_weights    : per-class weight, default equal. Applied as sample weights in
                           training, early stopping, tuning, calibration and every metric.
        feature_names    : default feature_0, feature_1, ... (no [ ] { } : , " allowed)
        """
        self._num_classes = num_classes

        self._params = {**DEFAULT_PARAMS, **(params or {}), "num_class": num_classes}
        if self._params["objective"] not in ("multiclass", "softmax"):
            _msg = "objective must be softmax cross-entropy ('multiclass')"
            raise ValueError(_msg)

        # --- class weighting (kept on the object for the equal-mix metrics) ---
        self._class_weights = (
            np.ones(num_classes)
            if class_weights is None
            else np.asarray(class_weights, dtype=np.float64)
        )

        # --- data, stored as (features, labels) pairs ---
        self._train = (train_features, np.asarray(train_labels, dtype=np.int64))
        self._val = (
            None
            if val_features is None
            else (val_features, np.asarray(val_labels, dtype=np.int64))
        )
        self._calib = (
            None
            if calib_features is None
            else (calib_features, np.asarray(calib_labels, dtype=np.int64))
        )

        self._feature_names = (
            list(feature_names)
            if feature_names is not None
            else [f"feature_{i}" for i in range(train_features.shape[1])]
        )

        self._num_boost_round = num_boost_round
        self._early_stopping_rounds = early_stopping_rounds
        self._log_period = log_period

        self._boosters: list[lgb.Booster] = []  # the currently loaded model(s)

    # ------------------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------------------
    def _sample_weights(self, labels):
        """Per-sample class weight, normalised to mean 1."""
        _w = self._class_weights[labels]
        return _w / _w.mean()

    def _fit(self, params, train_x, train_y, val_x, val_y, log=True):
        """One boosting run, early-stopped on the weighted val log-loss.
        Returns (booster, stats)."""
        _train_set = lgb.Dataset(
            train_x,
            label=train_y,
            weight=self._sample_weights(train_y),
            feature_name=self._feature_names,
        )
        _val_set = lgb.Dataset(
            val_x, label=val_y, weight=self._sample_weights(val_y), reference=_train_set
        )

        _evals = {}
        _callbacks = [lgb.record_evaluation(_evals)]
        if self._early_stopping_rounds is not None:
            _callbacks.append(
                lgb.early_stopping(
                    self._early_stopping_rounds, first_metric_only=True, verbose=False
                )
            )
        if log:
            _callbacks.append(lgb.log_evaluation(self._log_period))

        _booster = lgb.train(
            params,
            _train_set,
            num_boost_round=self._num_boost_round,
            valid_sets=[_train_set, _val_set],
            valid_names=["train", "val"],
            callbacks=_callbacks,
        )

        _best_round = _booster.best_iteration or _booster.current_iteration()
        _stats = {
            "best_round": _best_round,
            "best_val_loss": _evals["val"]["multi_logloss"][_best_round - 1],
            "train_losses": _evals["train"]["multi_logloss"],
            "val_losses": _evals["val"]["multi_logloss"],
            "train_accuracies": [1 - e for e in _evals["train"]["multi_error"]],
            "val_accuracies": [1 - e for e in _evals["val"]["multi_error"]],
        }
        return _booster, _stats

    def train(self, out_path):
        """Single model: fit on train, early-stop on val. Writes best_model.txt and
        training_stats.json (curves + the full params used)."""
        if self._val is None:
            _msg = "train() needs val data; use train_kfold() without it"
            raise ValueError(_msg)
        os.makedirs(out_path, exist_ok=True)  # noqa: PTH103

        _booster, _stats = self._fit(self._params, *self._train, *self._val)
        print(  # noqa: T201
            f"Best round {_stats['best_round']} - Val Loss: {_stats['best_val_loss']:.5f}"
        )

        # trees after the best round are dropped from the saved model
        _booster.save_model(
            f"{out_path}/best_model.txt", num_iteration=_stats["best_round"]
        )

        with open(f"{out_path}/training_stats.json", "w") as f:  # noqa: PTH123
            json.dump({**_stats, "params": self._params}, f)

        self._load(out_path)

    def train_kfold(self, out_path, folds):
        """One model per split you provide, each early-stopped on its own held-out part.

        folds : list of (train_idx, val_idx) index arrays into train_features/labels,
                e.g. list(StratifiedKFold(5, shuffle=True).split(X, y)). To use the val
                data too, build the object with train = train + val concatenated.

        Writes
          fold_XX/best_model.txt   one model per split; predictions average their logits
          oof.npz                  out-of-fold logits: each held-out sample predicted by
                                   the model(s) that did not train on it (averaged if a
                                   sample is held out in several splits)
          kfold_stats.json         per-fold curves, OOF log-loss, fold-to-fold spread
        """
        folds = list(folds)
        os.makedirs(out_path, exist_ok=True)  # noqa: PTH103
        _x, _y = self._train

        _oof_sum = np.zeros((len(_y), self._num_classes))
        _oof_count = np.zeros(len(_y), dtype=np.int64)
        fold_stats = []

        for _i, (_tr, _va) in enumerate(folds):
            print(f"--- fold {_i + 1}/{len(folds)}")  # noqa: T201
            _booster, _stats = self._fit(
                self._params, _x[_tr], _y[_tr], _x[_va], _y[_va]
            )
            print(  # noqa: T201
                f"Best round {_stats['best_round']} - Val Loss: {_stats['best_val_loss']:.5f}"
            )

            _fold_dir = f"{out_path}/fold_{_i:02d}"
            os.makedirs(_fold_dir, exist_ok=True)  # noqa: PTH103
            _booster.save_model(
                f"{_fold_dir}/best_model.txt", num_iteration=_stats["best_round"]
            )

            _oof_sum[_va] += _booster.predict(
                _x[_va], raw_score=True, num_iteration=_stats["best_round"]
            )
            _oof_count[_va] += 1
            fold_stats.append(_stats)

        _held = np.flatnonzero(_oof_count > 0)
        _oof = _oof_sum[_held] / _oof_count[_held, None]
        _oof_loss = self._weighted_nll(_oof, _y[_held])

        _fold_losses = np.array([s["best_val_loss"] for s in fold_stats])
        _fold_std = float(_fold_losses.std(ddof=1)) if len(folds) > 1 else 0.0
        print(  # noqa: T201
            f"OOF Loss: {_oof_loss:.5f} - folds {_fold_losses.mean():.5f} +- {_fold_std:.5f}"
        )

        np.savez(f"{out_path}/oof.npz", logits=_oof, labels=_y[_held], index=_held)
        with open(f"{out_path}/kfold_stats.json", "w") as f:  # noqa: PTH123
            json.dump(
                {
                    "n_folds": len(folds),
                    "n_oof_samples": len(_held),
                    "oof_log_loss": _oof_loss,
                    "fold_loss_mean": float(_fold_losses.mean()),
                    "fold_loss_std": _fold_std,
                    "folds": fold_stats,
                    "params": self._params,
                },
                f,
            )

        self._load(out_path)

    # ------------------------------------------------------------------------------
    # hyperparameter search
    # ------------------------------------------------------------------------------
    def tune(
        self,
        out_path,
        n_trials=50,
        folds=None,
        learning_rate=0.1,
        search_space=None,
        timeout=None,
    ):
        """Optuna (TPE) search minimising the equal-mix val log-loss at the early-stopped
        round. The best values are merged into this object's params, so a following
        train() / train_kfold() uses them.

        folds          : None -> score on the val data. A list of (train_idx, val_idx)
                         like train_kfold -> mean loss over the splits (slower, steadier).
        learning_rate  : used during the search only (faster trials); the tree settings
                         carry over to your own, smaller final rate.
        search_space   : default DEFAULT_SEARCH_SPACE.

        Writes tuning.json:
          best_params  the searched values of the best trial
          params       the full params now in use (defaults + your params + best_params,
                       with YOUR learning rate) -> GBTreeTraining(..., params=this)
          trials       every trial's values and score
        """
        import optuna  # noqa: PLC0415

        if folds is None and self._val is None:
            _msg = "tune() needs val data, or pass folds"
            raise ValueError(_msg)
        folds = None if folds is None else list(folds)
        os.makedirs(out_path, exist_ok=True)  # noqa: PTH103
        _space = search_space or DEFAULT_SEARCH_SPACE
        _x, _y = self._train

        def _objective(trial):
            _params = {**self._params, "learning_rate": learning_rate}
            for _name, _spec in _space.items():
                if _spec[0] == "int":
                    _params[_name] = trial.suggest_int(_name, *_spec[1:3], log=_spec[3])
                elif _spec[0] == "float":
                    _params[_name] = trial.suggest_float(
                        _name, *_spec[1:3], log=_spec[3]
                    )
                else:
                    _params[_name] = trial.suggest_categorical(_name, _spec[1])

            if folds is None:
                _, _stats = self._fit(_params, *self._train, *self._val, log=False)
                trial.set_user_attr("best_round", _stats["best_round"])
                return _stats["best_val_loss"]

            _losses = []
            for _tr, _va in folds:
                _, _stats = self._fit(
                    _params, _x[_tr], _y[_tr], _x[_va], _y[_va], log=False
                )
                _losses.append(_stats["best_val_loss"])
            return float(np.mean(_losses))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self._params["seed"]),
        )
        study.optimize(
            _objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True
        )

        self._params.update(study.best_params)
        print(f"Best val loss {study.best_value:.5f} (trial {study.best_trial.number})")  # noqa: T201

        with open(f"{out_path}/tuning.json", "w") as f:  # noqa: PTH123
            json.dump(
                {
                    "best_value": study.best_value,
                    "best_params": study.best_params,
                    "search_learning_rate": learning_rate,
                    "params": self._params,
                    "trials": [
                        {"number": t.number, "value": t.value, "params": t.params}
                        for t in study.trials
                    ],
                },
                f,
                indent=2,
            )

        return study.best_params

    # ------------------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------------------
    def _load(self, model_path):
        """Model file, a train() out_path, or a train_kfold() out_path (all folds)."""
        if os.path.isdir(model_path):  # noqa: PTH112
            _files = glob.glob(f"{model_path}/best_model.txt") or sorted(  # noqa: PTH207
                glob.glob(f"{model_path}/fold_*/best_model.txt")  # noqa: PTH207
            )
            if not _files:
                _msg = f"no best_model.txt under {model_path}"
                raise FileNotFoundError(_msg)
        else:
            _files = [model_path]
        self._boosters = [lgb.Booster(model_file=f) for f in _files]

    def _logits(self, features):
        """Raw scores (summed tree outputs per class, before softmax), averaged over the
        loaded model(s)."""
        return np.mean(
            [b.predict(features, raw_score=True) for b in self._boosters], axis=0
        )

    def _weighted_nll(self, logits, labels, temperature=1.0):
        """Class-weighted mean cross-entropy, same as nn.CrossEntropyLoss(weight=...)."""
        _nll = -log_softmax(logits / temperature, axis=1)[
            np.arange(len(labels)), labels
        ]
        return float(np.average(_nll, weights=self._class_weights[labels]))

    # ------------------------------------------------------------------------------
    # calibration
    # ------------------------------------------------------------------------------
    def fit_temperature(self, model_path, out_path=None, bounds=(0.01, 100.0)):
        """Single temperature T minimising the equal-mix log-loss of logits / T.
        T > 1 softens over-confident predictions, T < 1 sharpens them. Argmax (and so
        accuracy / confusion matrix / AUC) is unchanged.

        Fitted on the first available of
          calib  calib data: unseen by training and early stopping -> unbiased
          oof    out-of-fold logits of a train_kfold() model. Unbiased, but each comes
                 from one fold model while predictions average all folds (a bit less
                 confident), so T may come out slightly too large for the ensemble
          val    val data, which also chose the early-stopping round -> slightly
                 optimistic (warns)
        """
        self._load(model_path)
        _oof_file = f"{model_path}/oof.npz"

        if self._calib is not None:
            _source = "calib"
            _logits, _labels = self._logits(self._calib[0]), self._calib[1]
        elif len(self._boosters) > 1 and os.path.isfile(_oof_file):  # noqa: PTH113
            _source = "oof"
            _oof = np.load(_oof_file)
            _logits, _labels = _oof["logits"], _oof["labels"]
        elif self._val is not None:
            _source = "val"
            warnings.warn(
                "fitting the temperature on val data, which also chose the early-stopping "
                "round; pass calib data for an unbiased fit",
                stacklevel=2,
            )
            _logits, _labels = self._logits(self._val[0]), self._val[1]
        else:
            _msg = "no data to fit the temperature on; pass calib data or use train_kfold()"
            raise ValueError(_msg)

        def _objective(log_t):
            return self._weighted_nll(_logits, _labels, temperature=np.exp(log_t))

        _res = minimize_scalar(_objective, bounds=np.log(bounds), method="bounded")
        temperature = float(np.exp(_res.x))

        if out_path is not None:
            with open(f"{out_path}/temperature.json", "w") as f:  # noqa: PTH123
                json.dump(
                    {
                        "temperature": temperature,
                        "source": _source,
                        "n_samples": len(_labels),
                        "log_loss": _objective(0.0),
                        "log_loss_fitted": _objective(_res.x),
                    },
                    f,
                )

        return temperature

    # ------------------------------------------------------------------------------
    # equal-mix metrics
    # ------------------------------------------------------------------------------
    def log_loss_report(self, labels, probs):
        """Equal-mix log-loss: class-weighted mean of -ln(probability given to the truth).

        log_loss   : lower is better, 0 = perfect
        baseline   : ln K, the score of a model that ignores the features (1/K each)
        info_bits  : (baseline - log_loss) / ln 2, environment information captured
        pseudo_R2  : 1 - log_loss / baseline, fraction of the uncertainty removed
        typical_p  : exp(-log_loss), typical probability given to the right answer
        per_class  : mean -ln p_true within each true class
        """
        labels = np.asarray(labels)
        _w = self._class_weights[labels]
        _nll = -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0))

        _ll = float(np.average(_nll, weights=_w))
        _base = float(np.log(self._num_classes))

        return {
            "log_loss": _ll,
            "baseline": _base,
            "info_bits": (_base - _ll) / np.log(2),
            "pseudo_R2": 1 - _ll / _base,
            "typical_p": float(np.exp(-_ll)),
            "per_class": {
                k: float(_nll[labels == k].mean()) for k in range(self._num_classes)
            },
        }

    def ovr_auc(self, labels, probs):
        """One-vs-rest AUC per class: probability that a random true-k galaxy gets a higher
        P(k) than a random non-k galaxy (other classes weighted equally). 0.5 = no
        information, 1 = perfect ranking."""
        labels = np.asarray(labels)
        _w = self._class_weights[labels]

        aucs = {}
        roc_curves = {}
        for k in range(self._num_classes):
            _is_k = (labels == k).astype(int)
            aucs[k] = float(roc_auc_score(_is_k, probs[:, k], sample_weight=_w))

            _fpr, _tpr, _ = roc_curve(_is_k, probs[:, k], sample_weight=_w)
            _grid = np.linspace(0.0, 1.0, 201)
            roc_curves[k] = (_grid.tolist(), np.interp(_grid, _fpr, _tpr).tolist())

        return {
            "aucs": aucs,
            "roc_curves": roc_curves,
        }

    # ------------------------------------------------------------------------------
    # evaluation / prediction
    # ------------------------------------------------------------------------------
    def test(self, features, labels, model_path, temperature=1.0, out_path=None):
        """Confusion matrix (argmax, row-normalised = completeness), equal-mix log-loss
        and one-vs-rest AUC. Returns (labels, probabilities) for plotting."""
        self._load(model_path)
        labels = np.asarray(labels, dtype=np.int64)

        _logits = self._logits(features)
        _confusion_matrix = confusion_matrix(
            labels,
            _logits.argmax(axis=1),
            labels=list(range(self._num_classes)),
            normalize="true",
        )
        probs = softmax(_logits / temperature, axis=1)

        if out_path is not None:
            with open(f"{out_path}/test_metrics.json", "w") as f:  # noqa: PTH123
                json.dump(
                    {
                        "temperature": temperature,
                        "n_models": len(self._boosters),
                        "confusion_matrix": _confusion_matrix.tolist(),
                        "log_loss": self.log_loss_report(labels, probs),
                        "ovr_auc": self.ovr_auc(labels, probs),
                    },
                    f,
                )

        return labels, probs

    def predict(self, features, model_path, temperature=1.0, return_folds=False):
        """Equal-mix probabilities, shape (n, K).

        return_folds : also return each model's probabilities, shape (n_models, n, K).
                     Use these for the uncertainty of any statistic computed from many
                     galaxies (e.g. a mean in a bin): compute the statistic once per fold
                     and take the std over folds. Per-galaxy stds cannot be combined
                     as if independent, because one model's errors are correlated
                     across galaxies.
        Returns probs, followed by fold probabilities if requested.
        """
        self._load(model_path)
        probs = softmax(self._logits(features) / temperature, axis=1)
        if not return_folds:
            return probs

        if len(self._boosters) < 2:
            _msg = (
                "return_std / return_folds need several models (a train_kfold() folder)"
            )
            raise ValueError(_msg)
        _fold_probs = np.stack(
            [
                softmax(b.predict(features, raw_score=True) / temperature, axis=1)
                for b in self._boosters
            ]
        )  # (n_models, n, K)

        return (probs, _fold_probs)

    # ------------------------------------------------------------------------------
    # SHAP interpretability
    # ------------------------------------------------------------------------------
    def shap_values(
        self,
        features,
        model_path,
        labels=None,
        temperature=1.0,
        feature_groups=None,
        out_path=None,
    ):
        """Exact TreeSHAP values from LightGBM itself (pred_contrib), in logit space.

        values[i, k, j] : contribution of feature j to the (temperature-scaled) logit of
                          class k for sample i
        base[i, k]      : expected logit of class k
        additivity      : base + values.sum(-1) == logits / T, exactly (for k folds the
                          fold SHAP values are averaged = SHAP of the averaged logits)

        Summary per group (a group's SHAP is the sum over its columns; keep correlated
        features, e.g. a flag and its continuous partner, in one group), sorted by
        mean_abs:
          mean_abs                equal-mix mean |SHAP|, averaged over the class logits
          mean_abs_frac           mean_abs / sum of mean_abs over the single features
                                  (a group scores below the sum of its members when
                                  their contributions cancel within a sample)
          per_class_mean_abs      equal-mix mean |SHAP| on each class's logit
          toward_truth            mean of phi[i, y_i] - mean_k phi[i, k]: how much the
                                  feature pushes samples toward their true class relative
                                  to the others (> 0 helps, < 0 misleads; needs labels)
          per_class_toward_truth  the same within each true class (needs labels)

        For shap's plots: shap.summary_plot(values[:, k, :], features, feature_names=names)
        Writes shap_values.npz and shap_summary.json.
        """
        self._load(model_path)
        _names = self._boosters[0].feature_name()
        _n, _f, _k = len(features), len(_names), self._num_classes

        # LightGBM layout per row: for each class [feature_0 ... feature_{F-1}, bias]
        _contrib = (
            np.mean(
                [
                    b.predict(features, pred_contrib=True).reshape(_n, _k, _f + 1)
                    for b in self._boosters
                ],
                axis=0,
            )
            / temperature
        )
        values, base = _contrib[..., :-1], _contrib[..., -1]

        # --- summary ---
        if labels is not None:
            labels = np.asarray(labels, dtype=np.int64)
        _w = self._class_weights[labels] if labels is not None else np.ones(_n)
        if feature_groups is None:
            feature_groups = {_name: [_j] for _j, _name in enumerate(_names)}

        _total = float(
            np.average(np.abs(values), axis=0, weights=_w).mean(axis=0).sum()
        )

        groups = {}
        for _name, _cols in feature_groups.items():
            _idx = [_names.index(c) if isinstance(c, str) else int(c) for c in _cols]
            _phi = values[:, :, _idx].sum(axis=2)  # (n, K)
            _per_class_abs = np.average(np.abs(_phi), axis=0, weights=_w)

            groups[_name] = {
                "columns": list(_cols),
                "mean_abs": float(_per_class_abs.mean()),
                "mean_abs_frac": float(_per_class_abs.mean() / _total),
                "per_class_mean_abs": _per_class_abs.tolist(),
            }
            if labels is not None:
                _toward = _phi[np.arange(_n), labels] - _phi.mean(axis=1)
                groups[_name]["toward_truth"] = float(np.average(_toward, weights=_w))
                groups[_name]["per_class_toward_truth"] = [
                    float(_toward[labels == k].mean()) if np.any(labels == k) else None
                    for k in range(_k)
                ]

        summary = {
            "temperature": temperature,
            "n_models": len(self._boosters),
            "n_samples": _n,
            "base_logits": base.mean(axis=0).tolist(),
            "groups": dict(sorted(groups.items(), key=lambda kv: -kv[1]["mean_abs"])),
        }

        if out_path is not None:
            os.makedirs(out_path, exist_ok=True)  # noqa: PTH103
            np.savez_compressed(
                f"{out_path}/shap_values.npz",
                values=values,
                base=base,
                labels=labels if labels is not None else np.array([], dtype=np.int64),
                feature_names=np.array(_names),
            )
            with open(f"{out_path}/shap_summary.json", "w") as f:  # noqa: PTH123
                json.dump(summary, f, indent=2)

        return {
            "values": values,
            "base": base,
            "labels": labels,
            "feature_names": _names,
            "summary": summary,
        }
