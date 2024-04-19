from typing import Union, List
from collections import namedtuple, OrderedDict

import torch.nn as nn


_DenseFeatureSchema = namedtuple(
    typename="DenseFeature",
    field_names=["name", "dim", "dtype"]
)
_SparseFeatureSchema = namedtuple(
    typename="SparseFeature",
    field_names=["name", "vocabulary_size", "embedding_dim", "dtype", "embedding_name"]
)
_VarLenSparseFeatureSchema = namedtuple(
    typename="VarLenSparseFeature",
    field_names=["sparse_feature", "max_len", "combiner", "length_field"]
)


class DenseFeature(_DenseFeatureSchema):
    __slots__ = ()

    def __new__(cls, name: str, dim: int = 1, dtype: str = "float32"):
        return super().__new__(cls, name, dim, dtype)


class SparseFeature(_SparseFeatureSchema):
    __slots__ = ()

    def __new__(cls, name: str, vocabulary_size: int, embedding_dim: int, dtype: str = "int32", embedding_name: str = None):
        embedding_name = name if embedding_name is None else embedding_name

        return super().__new__(cls, name, vocabulary_size, embedding_dim, dtype, embedding_name)


class VarLenSparseFeature(_VarLenSparseFeatureSchema):
    __slots__ = ()

    def __new__(cls, sparse_feature: SparseFeature, max_len: int, combiner: str = "mean", length_field: str = None):
        return super().__new__(cls, sparse_feature, max_len, combiner, length_field)

    @property
    def name(self):
        return self.sparse_feature.name

    @property
    def vocabulary_size(self):
        return self.sparse_feature.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparse_feature.embedding_dim

    @property
    def dtype(self):
        return self.sparse_feature.dtype

    @property
    def embedding_name(self):
        return self.sparse_feature.embedding_name


Feature = Union[SparseFeature, DenseFeature]


def build_feature_offset(features: List[Feature]) -> OrderedDict:
    feature_offset = OrderedDict()

    start = 0
    for feature in features:
        if isinstance(feature, SparseFeature):
            feature_offset[feature.name] = (start, start + 1)
            start += 1

        elif isinstance(feature, DenseFeature):
            feature_offset[feature.name] = (start, start + feature.dim)
            start += feature.dim

        else:
            raise TypeError(f"Invalid feature column type. Got {type(feature)}")

    return feature_offset


def create_embedding_matrix(sparse_features: List[SparseFeature], linear: bool = False):
    embedding_dict = nn.ModuleDict({
        feature.name: nn.Embedding(
            num_embeddings=feature.vocabulary_size,
            embedding_dim=feature.embedding_dim if not linear else 1
        )
        for feature in sparse_features
    })

    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=1e-4)

    return embedding_dict
