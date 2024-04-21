from typing import List

import torch
import torch.nn as nn

from model.inputs import Feature, create_embedding_matrix
from model.architecture._model_object import _ModelObject


class Linear(_ModelObject):
    def __init__(self, linear_features: List[Feature], device: str = "cpu"):
        super().__init__(linear_features)

        self.device = device

        self.sparse_weights = None
        self.dense_weights = None
        self._initialize_weights()

    def _initialize_weights(self):
        if self.use_sparse_feature:
            self.sparse_weights = create_embedding_matrix(self.sparse_features, linear=True).to(self.device)

        if self.use_dense_feature:
            dim = sum(feature.dim for feature in self.dense_features)
            self.dense_weights = nn.Parameter(torch.Tensor(dim, 1)).to(self.device)
            nn.init.normal_(self.dense_weights, mean=0, std=1e-4)

    def forward(self, x: torch.Tensor):
        logit = torch.zeros([x.shape[0], 1]).to(self.device)

        if self.use_sparse_feature:
            values = self.get_sparse_embedding_values(x, self.sparse_weights)
            logit += torch.sum(torch.cat(values, dim=-1), dim=-1, keepdim=False)

        if self.use_dense_feature:
            values = self.get_dense_values(x)
            logit += torch.cat(values, dim=-1).matmul(self.dense_weights)

        return logit


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    from model.inputs import SparseFeature, DenseFeature

    DATASET_PATH = "/Users/youngho/Documents/GitHub/click-like/data/prep_test.csv"

    TARGET = ["label"]
    SPARSE_FEATURES = [f"C{index}" for index in range(1, 27)]
    DENSE_FEATURES = [f"I{index}" for index in range(1, 14)]

    data_frame = pd.read_csv(DATASET_PATH)

    sparse_feature_columns = [
        SparseFeature(feature, vocabulary_size=data_frame[feature].max() + 1, embedding_dim=4)
        for feature in SPARSE_FEATURES
    ]
    dense_feature_columns = [
        DenseFeature(feature, dim=1, )
        for feature in DENSE_FEATURES
    ]
    features = sparse_feature_columns + dense_feature_columns

    train_inputs = {
        feature.name: data_frame[feature.name]
        for feature in features
    }

    model = Linear(features)

    x = [train_inputs[feature_name] for feature_name in model.feature_offset.keys()]

    for index in range(0, len(x)):
        if len(x[index].shape) == 1:
            x[index] = np.expand_dims(x[index], axis=1)

    x = torch.from_numpy(np.concatenate(x, axis=-1)).float()
    print(model(x))

