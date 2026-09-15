import json
import os

import torch
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

        # --- class weighting ---
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32).to(
                self._device
            )
            criterion = criterion.__class__(weight=class_weights)

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

    def test(self, test_loader, model_path, out_path=None):
        _checkpoint = torch.load(model_path, map_location=self._device)
        self._model.load_state_dict(_checkpoint["model_state_dict"])

        _confusion_matrix_metric = ConfusionMatrix(
            task="multiclass",
            num_classes=self._num_classes,
            normalize="true",
        ).to(self._device)

        self._model.eval()
        with torch.no_grad():
            for _inputs, _targets in test_loader:
                _inputs = _inputs.to(self._device)
                _targets = _targets.to(self._device)

                _outputs = self._model(_inputs)

                _confusion_matrix_metric.update(_outputs, _targets)

        confusion_matrix = _confusion_matrix_metric.compute().cpu().numpy()

        if out_path is not None:
            with open(f"{out_path}/test_results.json", "w") as f:  # noqa: PTH123
                json.dump(confusion_matrix.tolist(), f)

    def check_feature_importance(
        self, test_loader, model_path, out_path=None, n_repeats=1, seed=42
    ):
        _checkpoint = torch.load(model_path, map_location=self._device)
        self._model.load_state_dict(_checkpoint["model_state_dict"])

        torch.manual_seed(seed)

        accuracy_matrix = []

        _n_features = test_loader.dataset.tensors[0].shape[1]

        for i in tqdm(range(_n_features), desc="Feature Importance"):
            _avg_per_class_accuracy = 0

            for _ in range(n_repeats):
                _per_class_metric = Accuracy(
                    task="multiclass",
                    num_classes=self._num_classes,
                    average="none",
                ).to(self._device)

                self._model.eval()
                with torch.no_grad():
                    for _inputs, _targets in test_loader:
                        _inputs = _inputs.to(self._device).clone()
                        _targets = _targets.to(self._device)

                        _inputs[:, i] = _inputs[torch.randperm(_inputs.size(0)), i]

                        _outputs = self._model(_inputs)

                        _per_class_metric.update(_outputs, _targets)

                _per_class_accuracy = _per_class_metric.compute().cpu().numpy()
                _avg_per_class_accuracy += _per_class_accuracy

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
