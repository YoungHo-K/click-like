from typing import List

import torch
import torch.nn as nn

from model.architecture._model_object import ModelObject
from model.inputs import Feature, create_embedding_matrix


class Linear(ModelObject):
    def __init__(self, features: List[Feature], device: str = "cpu"):
        super().__init__(features, device)

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
