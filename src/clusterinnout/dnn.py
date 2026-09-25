import json
import os

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, ConfusionMatrix
from tqdm import tqdm


class SupervisedTraining:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        lr: float,
        num_classes: int,
        criterion=nn.CrossEntropyLoss(),  # noqa: B008
        optimizer=optim.Adam,
        optimizer_params=None,
        scheduler=optim.lr_scheduler.CosineAnnealingLR,
        scheduler_params=None,
        class_weights=None,
        device="mps",
    ):

        self._device = device
        self._num_classes = num_classes

        self._model = model.to(self._device, dtype=torch.float32)

        # --- class weighting (kept on the object for the equal-mix metrics) ---
        self._class_weights = (
            np.array([1.0] * num_classes, dtype=np.float32)
            if class_weights is None
            else np.asarray(class_weights, dtype=np.float32)
        )
        if self._class_weights is not None:
            criterion = criterion.__class__(
                weight=torch.as_tensor(self._class_weights, dtype=torch.float32)
            )

        self._criterion = criterion.to(self._device, dtype=torch.float32)

        self._optimizer = optimizer(
            self._model.parameters(), lr=lr, **(optimizer_params or {})
        )

        # --- CosineAnnealingLR scheduling ---
        if scheduler_params is None:
            scheduler_params = {}
        if (
            scheduler is optim.lr_scheduler.CosineAnnealingLR
            and "T_max" not in scheduler_params
        ):
            scheduler_params = {**scheduler_params, "T_max": num_epochs}

        self._scheduler = (
            scheduler(self._optimizer, **scheduler_params)
            if scheduler is not None
            else None
        )

        self._num_epochs = num_epochs
        self._train_loader = train_loader
        self._val_loader = val_loader

        self._accuracy_metric = Accuracy(
            task="multiclass", num_classes=self._num_classes
        ).to(self._device)

    # ------------------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------------------
    def _get_accuracy(self, outputs, targets):
        _preds = torch.argmax(outputs, dim=1)
        return self._accuracy_metric(_preds, targets).item()

    def _train_epoch(self):
        current_loss = 0.0
        current_accuracy = 0.0

        self._model.train()
        for _inputs, _targets in self._train_loader:
            _inputs = _inputs.to(self._device)
            _targets = _targets.to(self._device)

            self._optimizer.zero_grad()

            _outputs = self._model(_inputs)
            _loss = self._criterion(_outputs, _targets)

            current_loss += _loss.item()
            current_accuracy += self._get_accuracy(_outputs, _targets)

            _loss.backward()
            self._optimizer.step()

        if self._scheduler is not None:
            self._scheduler.step()

        n = len(self._train_loader)
        return current_loss / n, current_accuracy / n

    def _val_epoch(self):
        current_loss = 0.0
        current_accuracy = 0.0

        self._model.eval()
        with torch.no_grad():
            for _inputs, _targets in self._val_loader:
                _inputs = _inputs.to(self._device)
                _targets = _targets.to(self._device)

                _outputs = self._model(_inputs)
                _loss = self._criterion(_outputs, _targets)

                current_loss += _loss.item()
                current_accuracy += self._get_accuracy(_outputs, _targets)

        n = len(self._val_loader)
        return current_loss / n, current_accuracy / n

    def _save_model(self, out_path):
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
            },
            out_path,
        )

    def train(self, out_path):
        os.makedirs(out_path, exist_ok=True)  # noqa: PTH103

        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        _best_val_loss = float("inf")

        for _epoch in tqdm(range(1, self._num_epochs + 1), desc="Training"):
            _train_loss, _train_acc = self._train_epoch()
            _val_loss, _val_acc = self._val_epoch()

            if _val_loss < _best_val_loss:
                _best_val_loss = _val_loss
                self._save_model(f"{out_path}/best_model.pth")

            if _epoch % 10 == 0 or _epoch == 1:
                tqdm.write(
                    f"Epoch {_epoch}/{self._num_epochs} - Train Loss: {_train_loss:.5f} - Val Loss: {_val_loss:.5f}"
                )

            train_losses.append(_train_loss)
            val_losses.append(_val_loss)
            train_accuracies.append(_train_acc)
            val_accuracies.append(_val_acc)

        with open(f"{out_path}/training_stats.json", "w") as f:  # noqa: PTH123
            json.dump(
                {
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_accuracies": train_accuracies,
                    "val_accuracies": val_accuracies,
                },
                f,
            )

    # ------------------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------------------
    def _load(self, model_path):
        _checkpoint = torch.load(model_path, map_location=self._device)
        self._model.load_state_dict(_checkpoint["model_state_dict"])
        self._model.eval()

    def _collect_logits(self, loader):
        _logits, _targets = [], []
        with torch.no_grad():
            for _batch in loader:
                if isinstance(_batch, (list, tuple)):
                    _inputs = _batch[0]
                    if len(_batch) > 1:
                        _targets.append(_batch[1].cpu())
                else:
                    _inputs = _batch
                _logits.append(self._model(_inputs.to(self._device)).cpu())
        return torch.cat(_logits), (torch.cat(_targets) if _targets else None)

    def _to_probabilities(self, logits, temperature=1.0):
        return torch.softmax(logits / temperature, dim=1).numpy()

    # ------------------------------------------------------------------------------
    # calibration
    # ------------------------------------------------------------------------------
    def fit_temperature(self, model_path, out_path=None, bounds=(0.01, 100.0)):
        self._load(model_path)

        _logits, _targets = self._collect_logits(self._val_loader)
        _nll = nn.CrossEntropyLoss(
            weight=torch.as_tensor(self._class_weights, dtype=torch.float32)
        )

        def _objective(log_t):
            return _nll(_logits / np.exp(log_t), _targets).item()

        _res = minimize_scalar(_objective, bounds=np.log(bounds), method="bounded")
        temperature = float(np.exp(_res.x))

        if out_path is not None:
            with open(f"{out_path}/temperature.json", "w") as f:  # noqa: PTH123
                json.dump(
                    {
                        "temperature": temperature,
                        "val_log_loss": _objective(0.0),
                        "val_log_loss_fitted": _objective(_res.x),
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
            "pseudo_R2": 1 - _ll / _base,
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
            roc_curves[k] = (_fpr.tolist(), _tpr.tolist())

        return {
            "aucs": aucs,
            "roc_curves": roc_curves,
        }

    # ------------------------------------------------------------------------------
    # evaluation / prediction
    # ------------------------------------------------------------------------------
    def test(self, test_loader, model_path, temperature=1.0, out_path=None):
        """Confusion matrix (argmax of the equal-mix probabilities, row-normalised =
        completeness), equal-mix log-loss and one-vs-rest AUC on the test set.

        Returns (labels, probabilities) for plotting."""
        self._load(model_path)

        _logits, _targets = self._collect_logits(test_loader)

        _confusion_matrix_metric = ConfusionMatrix(
            task="multiclass",
            num_classes=self._num_classes,
            normalize="true",
        )
        _confusion_matrix_metric.update(_logits, _targets)
        confusion_matrix = _confusion_matrix_metric.compute().numpy()

        labels = _targets.numpy()
        probs = self._to_probabilities(_logits, temperature=temperature)

        if out_path is not None:
            with open(f"{out_path}/test_metrics.json", "w") as f:  # noqa: PTH123
                json.dump(
                    {
                        "temperature": temperature,
                        "confusion_matrix": confusion_matrix.tolist(),
                        "log_loss": self.log_loss_report(labels, probs),
                        "ovr_auc": self.ovr_auc(labels, probs),
                    },
                    f,
                )

        return labels, probs

    def predict(self, input_loader, model_path, temperature=1.0):
        """Equal-mix probabilities for any loader (inputs only, or (inputs, labels))."""
        self._load(model_path)
        _logits, _ = self._collect_logits(input_loader)
        return self._to_probabilities(_logits, temperature=temperature)

    # ------------------------------------------------------------------------------
    # feature importance
    # ------------------------------------------------------------------------------
    def check_feature_importance(
        self, test_loader, model_path, out_path=None, n_repeats=1, seed=42
    ):
        _checkpoint = torch.load(model_path, map_location=self._device)
        self._model.load_state_dict(_checkpoint["model_state_dict"])
        self._model.eval()

        # Full test set, not batch by batch
        _features, _labels = test_loader.dataset.tensors
        _n_samples, _n_features = _features.shape
        _batch_size = test_loader.batch_size

        _generator = torch.Generator().manual_seed(seed)

        accuracy_matrix = []

        for i in tqdm(range(_n_features), desc="Feature Importance"):
            _avg_per_class_accuracy = 0

            for _ in range(n_repeats):
                _perm = torch.randperm(_n_samples, generator=_generator)
                _permuted = _features.clone()
                _permuted[:, i] = _features[_perm, i]

                _per_class_metric = Accuracy(
                    task="multiclass", num_classes=self._num_classes, average="none"
                ).to(self._device)

                with torch.no_grad():
                    for _start in range(0, _n_samples, _batch_size):
                        _inputs = _permuted[_start : _start + _batch_size].to(
                            self._device
                        )
                        _targets = _labels[_start : _start + _batch_size].to(
                            self._device
                        )
                        _per_class_metric.update(self._model(_inputs), _targets)

                _avg_per_class_accuracy += _per_class_metric.compute().cpu().numpy()

            _avg_per_class_accuracy /= n_repeats
            accuracy_matrix.append(_avg_per_class_accuracy.tolist())

        if out_path is not None:
            with open(f"{out_path}/feature_importance.json", "w") as f:  # noqa: PTH123
                json.dump(accuracy_matrix, f)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        dropout: float = 0.0,
        activation=nn.ReLU,
        norm=nn.BatchNorm1d,
    ):
        super().__init__()

        layer_dims = [input_dim, *hidden_dims, output_dim]
        layers = []

        for i in range(len(layer_dims) - 1):
            is_last = i == len(layer_dims) - 2

            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

            if not is_last:
                if norm is not None:
                    layers.append(norm(layer_dims[i + 1]))
                layers.append(activation())
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self._input_dim = input_dim

        self._feature_embeddings = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(input_dim)]
        )

        self._cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self._pos_embedding = nn.Parameter(torch.zeros(1, input_dim + 1, d_model))

        _encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self._encoder = nn.TransformerEncoder(_encoder_layer, num_layers=num_layers)

        self._dropout = nn.Dropout(dropout)
        self._classifier = nn.Linear(d_model, output_dim)

        nn.init.trunc_normal_(self._cls_token, std=0.02)
        nn.init.trunc_normal_(self._pos_embedding, std=0.02)

    def forward(self, x):
        _batch_size = x.shape[0]

        _tokens = torch.stack(
            [
                _embed(x[:, i].unsqueeze(-1))
                for i, _embed in enumerate(self._feature_embeddings)
            ],
            dim=1,
        )

        _cls_tokens = self._cls_token.expand(_batch_size, -1, -1)
        _tokens = torch.cat([_cls_tokens, _tokens], dim=1)

        _tokens = _tokens + self._pos_embedding

        _encoded = self._encoder(_tokens)

        _cls_output = _encoded[:, 0]
        _cls_output = self._dropout(_cls_output)

        return self._classifier(_cls_output)
