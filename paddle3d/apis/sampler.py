import math

import numpy as np
import paddle
from paddle.io import Sampler
from paddle.io import DistributedBatchSampler as _DistributedBatchSampler


# ============================================
# DistributedBatchSampler for multi_gpu_test
# we should disable shuffle and drop_last
# ============================================
class DistributedTestBatchSampler(_DistributedBatchSampler):
    def __init__(
        self,
        dataset,
        batch_size=1,
        num_replicas=None,
        rank=None,
        shuffle=False,
        drop_last=False,
    ):
        super().__init__(
            dataset, batch_size, num_replicas=num_replicas, rank=rank, shuffle=shuffle
        )

    def __iter__(self):
        if self.shuffle:
            assert False
        else:
            indices = np.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices = (indices * math.ceil(self.total_size / len(indices)))[
            : self.total_size
        ]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.local_rank : self.total_size : self.nranks]
        assert len(indices) == self.num_samples

        batch_indices = []
        for idx in indices:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices
