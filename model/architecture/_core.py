from typing import List

import torch
import torch.nn as nn

from model.inputs import Feature, SparseFeature, DenseFeature, build_feature_offset, create_embedding_matrix


class Linear(nn.Module):
    def __init__(self, linear_features: List[Feature], device: str = "cpu"):
        super().__init__()

        self.feature_offset = build_feature_offset(linear_features)
        self.dense_features = list(filter(lambda x: isinstance(x, DenseFeature), linear_features))
        self.sparse_features = list(filter(lambda x: isinstance(x, SparseFeature), linear_features))
        self.device = device

        self.sparse_feature_weights = None
        self.dense_feature_weights = None
        self._initialize_weights()

    def _initialize_weights(self):
        if len(self.sparse_features) > 0:
            self.sparse_feature_weights = create_embedding_matrix(self.sparse_features, linear=True).to(self.device)

        if len(self.dense_features) > 0:
            dense_dim = sum(feature.dim for feature in self.dense_features)
            self.dense_feature_weights = nn.Parameter(torch.Tensor(dense_dim, 1)).to(self.device)
            nn.init.normal_(self.dense_feature_weights, mean=0, std=1e-4)

    def forward(self, x: torch.Tensor):
        logit = torch.zeros([x.shape[0], 1]).to(self.device)

        if self.sparse_feature_weights is not None:
            weights = [
                self.sparse_feature_weights[feature.name](
                    x[:, self.feature_offset[feature.name][0]: self.feature_offset[feature.name][1]]
                )
                for feature in self.sparse_feature_weights
            ]
            logit += torch.sum(torch.cat(weights, dim=-1), dim=-1, keepdim=False)

        if self.dense_feature_weights is not None:
            dense_values = [
                x[: self.feature_offset[feature.name][0]: self.feature_offset[feature.name][1]]
                for feature in self.dense_features
            ]

            logit += torch.cat(dense_values, dim=-1).matmul(self.dense_feature_weights)

        return logit
