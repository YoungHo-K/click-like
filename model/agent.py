import time
from typing import Optional, List, Tuple
from collections import defaultdict

import numpy as np
import tqdm
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error

from model.architecture import ModelObject


def get_optimizer(name: str):
    if name == "sgd":
        return optim.SGD

    if name == "adam":
        return optim.Adam

    if name == "adagrad":
        return optim.Adagrad

    if name == "rmsprop":
        return optim.RMSprop

    raise NotImplementedError


def get_criterion(name: str):
    if name == "binary_crossentropy":
        return F.binary_cross_entropy

    if name == "mse":
        return F.mse_loss

    if name == "mae":
        return F.l1_loss

    raise NotImplementedError


def get_metrics(metrics: List[str]) -> dict:
    metric_dict = {}
    for name in metrics:
        if (name == "binary_crossentropy") or (name == "logloss"):
            metric_dict[name] = log_loss

        elif name == "auc":
            metric_dict[name] = roc_auc_score

        elif name == "mse":
            metric_dict[name] = mean_squared_error

        else:
            print(f"NotImplementedError: `{name}`")

    return metric_dict


class CTRAgent:
    def __init__(self, model: ModelObject, optimizer: str, criterion: str, metrics: List[str], device: str = "cpu"):
        self.model = model
        self.optimizer = get_optimizer(optimizer)(self.model.parameters())
        self.criterion = get_criterion(criterion)
        self.metrics = get_metrics(metrics)
        self.device = device

        self.output_func = torch.sigmoid

    def train_model(
            self,
            x: dict,
            y: np.ndarray,
            val_data: Optional[Tuple[dict, np.ndarray]] = None,
            epoch: int = 1,
            batch_size: int = 32,
    ):
        x, y = self.convert_inputs_to_tensor(x), torch.from_numpy(y)
        train_dataset = TensorDataset(x, y)
        train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True)

        num_train_samples = len(train_dataset)
        steps_per_epoch = (num_train_samples - 1) // batch_size + 1

        msg = ""
        msg += f"Train on {num_train_samples:,} samples, "
        msg += f"{steps_per_epoch} steps per epoch."
        print(msg)

        for iteration in range(1, epoch + 1):
            self.model.train()

            start_time = time.time()
            total_loss = 0.0
            epoch_logs = {}
            train_results = defaultdict(list)
            try:
                with tqdm.tqdm(enumerate(train_data_loader)) as processor:
                    for _, (features, targets) in processor:
                        features = features.float().to(self.device)
                        targets = targets.float().to(self.device)
                        preds = self.output_func(self.model(features))

                        loss = self.criterion(preds.squeeze(), targets.squeeze(), reduction="sum")
                        loss.backward()
                        self.optimizer.step()

                        total_loss += loss.items()

                        for name, func in self.metrics.items():
                            train_results[name].append(func(targets.cpu().data.numpy(), preds.cpu().data.numpy()))

            except KeyboardInterrupt:
                raise

            finally:
                processor.close()

            epoch_logs["loss"] = total_loss / num_train_samples
            for name, result in train_results.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if val_data is not None:
                x_val, y_val = val_data[0], val_data[1]
                eval_result = self.evaluate(x_val, y_val, batch_size)
                for name, result in eval_result.items():
                    epoch_logs[f"val_{name}"] = result

            epoch_time = time.time() - start_time
            print(f"Epoch {iteration}/{epoch}")

            log = f"{epoch_time}s - loss: {epoch_logs['loss']:.4f}"
            for name in self.metrics:
                log += " - " + name + f": {epoch_logs[name]:.4f}"

            if val_data is not None:
                log += f" - val_loss: {epoch_logs['val_loss']:.4f}"
                for name in self.metrics:
                    log += " - " + f"val_{name}" + f": {epoch_logs['val_' + name]:.4f}"

            print(log)

    def convert_inputs_to_tensor(self, x: dict):
        x = [x[feature] for feature in self.model.feature_offset]
        for index in range(0, len(x)):
            if len(x[index].shape) == 1:
                x[index] = np.expand_dims(x[index], axis=1)

        return torch.from_numpy(np.concatenate(x, axis=-1))

    def predict(self, x: dict, batch_size: int = 32):
        x = self.convert_inputs_to_tensor(x)

        dataset = TensorDataset(x)
        data_loader = DataLoader(dataset, batch_size, shuffle=False)

        self.model.eval()

        total_preds = []
        with torch.no_grad():
            for features in data_loader:
                features = features[0].float().to(self.device)
                preds = self.output_func(self.model(features))

                total_preds.append(preds.cpu().data.numpy())

        return np.concatenate(total_preds).astype("float64")

    def evaluate(self, x: dict, y: np.ndarray, batch_size: int = 32):
        preds = self.predict(x, batch_size)

        eval_result = {
            name: func(y, preds)
            for name, func in self.metrics.items()
        }

        return eval_result

