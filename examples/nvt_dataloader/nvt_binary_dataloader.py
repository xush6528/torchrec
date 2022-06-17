#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import concurrent
import json
import math
import os
import queue
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch.distributed as dist
import torch
from torch.utils import data as data_utils
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchrec.datasets.criteo import (
    CAT_FEATURE_COUNT,
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
)
from torchrec.datasets.utils import Batch
from torchrec.metrics.throughput import ThroughputMetric
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

NUM_EMBEDDINGS_PER_FEATURE = [
    45833188,
    36746,
    17245,
    7413,
    20243,
    3,
    7114,
    1441,
    62,
    29275261,
    1572176,
    345138,
    10,
    2209,
    11267,
    128,
    4,
    974,
    14,
    48937457,
    11316796,
    40094537,
    452104,
    12606,
    104,
    35,
]


def get_categorical_feature_type(size: int):
    """This function works both when max value and cardinality is passed.
    Consistency by the user is required"""
    types = (np.int8, np.int16, np.int32)

    for numpy_type in types:
        if size < np.iinfo(numpy_type).max:
            return numpy_type

    raise RuntimeError(
        f"Categorical feature of size {size} is too big for defined types"
    )


class ParametricDataset(Dataset):
    def __init__(
        self,
        binary_file_path: str,
        batch_size: int = 1,  # should be same as the pre-proc
        prefetch_depth: int = 100,
        drop_last_batch: bool = False,
        **kwargs,
    ):
        self._batch_size = batch_size

        self._categorical_types = [
            get_categorical_feature_type(num) for num in NUM_EMBEDDINGS_PER_FEATURE
        ]

        bytes_per_feature = {}
        for name in DEFAULT_INT_NAMES:
            bytes_per_feature[name] = np.dtype(np.float16).itemsize
        for name, categorical_type in zip(DEFAULT_CAT_NAMES, self._categorical_types):
            bytes_per_feature[name] = np.dtype(categorical_type).itemsize

        self._numerical_features_file = None
        self._label_file = None
        self._categorical_features_files = []

        self._numerical_bytes_per_batch = (
            bytes_per_feature[DEFAULT_INT_NAMES[0]]
            * len(DEFAULT_INT_NAMES)
            * batch_size
        )
        self._label_bytes_per_batch = np.dtype(np.bool).itemsize * batch_size
        self._categorical_bytes_per_batch = [
            bytes_per_feature[feature] * self._batch_size
            for feature in DEFAULT_CAT_NAMES
        ]
        # Load categorical
        for feature_name in DEFAULT_CAT_NAMES:
            path_to_open = os.path.join(binary_file_path, f"{feature_name}.bin")
            cat_file = os.open(path_to_open, os.O_RDONLY)
            bytes_per_batch = bytes_per_feature[feature_name] * self._batch_size
            batch_num_float = os.fstat(cat_file).st_size / bytes_per_batch
            self._categorical_features_files.append(cat_file)

        # Load numerical
        path_to_open = os.path.join(binary_file_path, "numerical.bin")
        self._numerical_features_file = os.open(path_to_open, os.O_RDONLY)
        batch_num_float = (
            os.fstat(self._numerical_features_file).st_size
            / self._numerical_bytes_per_batch
        )

        # Load label
        path_to_open = os.path.join(binary_file_path, "label.bin")
        self._label_file = os.open(path_to_open, os.O_RDONLY)
        batch_num_float = (
            os.fstat(self._label_file).st_size / self._label_bytes_per_batch
        )

        number_of_batches = (
            math.ceil(batch_num_float)
            if not drop_last_batch
            else math.floor(batch_num_float)
        )

        self._num_entries = number_of_batches
        self._prefetch_depth = min(prefetch_depth, self._num_entries)
        self._prefetch_queue = queue.Queue()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def __len__(self):
        return self._num_entries

    def __getitem__(self, idx: int):
        """Numerical features are returned in the order they appear in the channel spec section
        For performance reasons, this is required to be the order they are saved in, as specified
        by the relevant chunk in source spec.
        Categorical features are returned in the order they appear in the channel spec section"""

        if idx >= self._num_entries:
            raise IndexError()

        if self._prefetch_depth <= 1:
            return self._get_item(idx)

        # At the start, fill up the prefetching queue
        if idx == 0:
            for i in range(self._prefetch_depth):
                self._prefetch_queue.put(self._executor.submit(self._get_item, (i)))
        # Extend the prefetching window by one if not at the end of the dataset
        if idx < self._num_entries - self._prefetch_depth:
            self._prefetch_queue.put(
                self._executor.submit(self._get_item, (idx + self._prefetch_depth))
            )
        return self._prefetch_queue.get().result()

    def _get_item(
        self, idx: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        click = self._get_label(idx)
        numerical_features = self._get_numerical_features(idx)
        categorical_features = self._get_categorical_features(idx)
        return numerical_features, categorical_features, click

    def _get_label(self, idx: int) -> torch.Tensor:
        raw_label_data = os.pread(
            self._label_file,
            self._label_bytes_per_batch,
            idx * self._label_bytes_per_batch,
        )
        # array = np.frombuffer(raw_label_data, dtype=np.bool)
        # return torch.from_numpy(array).to(torch.float32)
        return torch.frombuffer(raw_label_data, dtype=torch.bool).to(torch.float32)

    def _get_numerical_features(self, idx: int) -> Optional[torch.Tensor]:
        if self._numerical_features_file is None:
            return None

        raw_numerical_data = os.pread(
            self._numerical_features_file,
            self._numerical_bytes_per_batch,
            idx * self._numerical_bytes_per_batch,
        )
        # array = np.frombuffer(raw_numerical_data, dtype=np.float16)
        # return torch.from_numpy(array).to(torch.float32).view(-1, len(DEFAULT_INT_NAMES))
        return (
            torch.frombuffer(raw_numerical_data, dtype=torch.float16)
            .to(torch.float32)
            .view(-1, len(DEFAULT_INT_NAMES))
        )

    def _convert_np_to_torch_type(self, np_type):
        if np_type == np.int8:
            return torch.int8
        elif np_type == np.int16:
            return torch.int16
        elif np_type == np.int32:
            return torch.int32
        return torch.int32

    def _get_categorical_features(self, idx: int) -> Optional[torch.Tensor]:
        if self._categorical_features_files is None:
            return None
        categorical_features = []
        for cat_bytes, cat_type, cat_file in zip(
            self._categorical_bytes_per_batch,
            self._categorical_types,
            self._categorical_features_files,
        ):
            raw_cat_data = os.pread(cat_file, cat_bytes, idx * cat_bytes)
            # array = np.frombuffer(raw_cat_data, dtype=cat_type)
            # tensor = torch.from_numpy(array).to(torch.long).view(-1)
            tensor = (
                torch.frombuffer(
                    raw_cat_data, dtype=self._convert_np_to_torch_type(cat_type)
                )
                .to(torch.long)
                .view(-1)
            )
            categorical_features.append(tensor)
        return torch.cat(categorical_features)

    # def __del__(self):
    #     data_files = [self._label_file, self._numerical_features_file]
    #     if self._categorical_features_files is not None:
    #         data_files += self._categorical_features_files

    #     for data_file in data_files:
    #         if data_file is not None:
    #             os.close(data_file)


class NvtBinaryDataloader:
    def __init__(
        self,
        binary_file_path: str,
        batch_size: int = 1,  # should be same as the pre-proc
        prefetch_depth: int = 1,
        drop_last_batch: bool = True,
    ) -> None:
        self.dataset = ParametricDataset(
            binary_file_path,
            batch_size,
            prefetch_depth,
            drop_last_batch,
        )
        self._num_ids_in_batch: int = CAT_FEATURE_COUNT * batch_size
        self.keys: List[str] = DEFAULT_CAT_NAMES
        self.lengths: torch.Tensor = torch.ones(
            (self._num_ids_in_batch,), dtype=torch.int32
        )
        self.offsets: torch.Tensor = torch.arange(
            0, self._num_ids_in_batch + 1, dtype=torch.int32
        )
        self.stride = batch_size
        self.length_per_key: List[int] = CAT_FEATURE_COUNT * [batch_size]
        self.offset_per_key: List[int] = [
            batch_size * i for i in range(CAT_FEATURE_COUNT + 1)
        ]
        self.index_per_key: Dict[str, int] = {
            key: i for (i, key) in enumerate(self.keys)
        }

    def collate_fn(self, attr_dict):
        dense_features, sparse_features, labels = attr_dict
        dense_features = dense_features.type(torch.FloatTensor)
        # We know that all categories are one-hot. However, this may not generalize
        # We should work with nvidia to allow nvtabular to natively transform to
        # a KJT format.
        return Batch(
            dense_features=dense_features,
            sparse_features=KeyedJaggedTensor(
                keys=DEFAULT_CAT_NAMES,
                values=sparse_features,
                lengths=self.lengths,
                offsets=self.offsets,
                stride=self.stride,
                length_per_key=self.length_per_key,
                offset_per_key=self.offset_per_key,
                index_per_key=self.index_per_key,
            ),
            labels=labels,
        )

    def get_dataloader(
        self,
        rank: int,
        world_size: int,
    ) -> data_utils.DataLoader:
        sampler = DistributedSampler(
            self.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        dataloader = data_utils.DataLoader(
            self.dataset,
            batch_size=None,
            pin_memory=False,
            collate_fn=self.collate_fn,
            sampler=sampler,
            num_workers=0,
        )
        return dataloader
