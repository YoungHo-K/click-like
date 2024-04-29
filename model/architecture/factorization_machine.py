from typing import List

import torch

from model.architecture._model_object import ModelObject
from model.inputs import SparseFeature, create_embedding_matrix


class FactorizationMachine(ModelObject):
    def __init__(self, sparse_features: List[SparseFeature], device: str = "cpu"):
        super().__init__(sparse_features, device)

        self.embedding_dict = create_embedding_matrix(self.sparse_features, linear=False).to(self.device)

    def forward(self, x: torch.Tensor):
        embedding_values = self.get_sparse_embedding_values(x, self.embedding_dict)
        embedding_values = torch.cat(embedding_values, dim=1)

        square_of_sum = torch.pow(torch.sum(embedding_values, dim=1), exponent=2)
        sum_of_square = torch.sum(torch.pow(embedding_values, exponent=2), dim=1)

        return torch.sum(square_of_sum - sum_of_square, dim=1) * 0.5
