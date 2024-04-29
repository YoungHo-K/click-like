from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from model.inputs import Feature, SparseFeature, DenseFeature, build_feature_offset


class ModelObject(nn.Module):
    def __init__(self, features: List[Feature], device: str = "cpu"):
        super().__init__()

        self.feature_offset = build_feature_offset(features)
        self.dense_features = list(filter(lambda x: isinstance(x, DenseFeature), features))
        self.sparse_features = list(filter(lambda x: isinstance(x, SparseFeature), features))

        self.device = device

    @property
    def use_sparse_feature(self):
        return len(self.sparse_features) > 0

    @property
    def use_dense_feature(self):
        return len(self.dense_features) > 0

    def get_sparse_embedding_values(self, x: torch.Tensor, embedding_dict: nn.ModuleDict) -> List[torch.Tensor]:
        embedding_values = [
            embedding_dict[feature.name](
                x[:, self.feature_offset[feature.name][0]: self.feature_offset[feature.name][1]].long()
            )
            for feature in self.sparse_features
        ]

        return embedding_values

    def get_dense_values(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [
            x[:, self.feature_offset[feature.name][0]: self.feature_offset[feature.name][1]]
            for feature in self.dense_features
        ]
