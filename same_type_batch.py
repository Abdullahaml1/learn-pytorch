import torch
from torch.utils.data.sampler import Sampler
import numpy as np
from typing import List, Iterable, Union, Iterator
from collections import defaultdict


def generate_data(size: 20) -> list[dict[str, int]]:
    """
    generate data at form
    [{'type': , (positive, or negative), 'val': (int)]
    ex: [{'type': 'pos', 'val': 3}, {'type': 'neg', 'val': 1}]
    """
    vals = np.random.randint(0, 10, size=size)
    rand = np.random.randn(size)

    out_list = []
    for val, rand in zip(vals, rand):
        data_type = 'pos' if rand > 0.5 else 'neg'
        item = {'val': val, 'type': data_type}
        out_list.append(item)

    types = [d['type'] for d in out_list]

    return out_list, types


class TypeBatchSampler(Sampler[List[int]]):
    """
    Rewritten from:
    https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#BatchSampler
    Read also: https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler
    """
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler: Union[Sampler[int], Iterable[int]],
                 batch_size: int,
                 data_types: List[str],
                 drop_last: bool = False) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.data_types = data_types

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        # if self.drop_last:
        #     sampler_iter = iter(self.sampler)
        #     while True:
        #         try:
        #             batch = [next(sampler_iter) for _ in range(self.batch_size)]
        #             yield batch
        #         except StopIteration:
        #             break
        # else:
        #     batch = [0] * self.batch_size
        #     idx_in_batch = 0
        #     for idx in self.sampler:
        #         batch[idx_in_batch] = idx
        #         idx_in_batch += 1
        #         if idx_in_batch == self.batch_size:
        #             yield batch
        #             idx_in_batch = 0
        #             batch = [0] * self.batch_size
        #     if idx_in_batch > 0:
        #         yield batch[:idx_in_batch]
        types_dict = defaultdict(lambda: [0] * self.batch_size)
        idx_in_batch_dict = defaultdict(lambda: 0)
        for idx in self.sampler:
            dtype = self.data_types[idx]
            types_dict[dtype][idx_in_batch_dict[dtype]] = idx
            idx_in_batch_dict[dtype] += 1
            if idx_in_batch_dict[dtype] == self.batch_size:
                yield types_dict[dtype]
                idx_in_batch_dict[dtype] = 0
                types_dict[dtype] = [0] * self.batch_size

        for dtype, ids in types_dict.items():
            idx_in_batch = idx_in_batch_dict[dtype]
            if idx_in_batch > 0:
                yield ids[:idx_in_batch]

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]


if __name__ == "__main__":

    # generating data
    data, types = generate_data(10)
    for x in data:
        print(x)
    print('------------')

    # generate batch with every type
    sampler = torch.utils.data.sampler.SequentialSampler(data)
    batch_sampler = TypeBatchSampler(sampler, batch_size=4, data_types=types)
    dataloader = torch.utils.data.DataLoader(data, batch_sampler=batch_sampler)

    for batch in dataloader:
        print(batch)
