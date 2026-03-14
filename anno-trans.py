import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    # Some imports
    import os
    from os.path import exists
    import torch
    import torch.nn as nn
    from torch.nn.functional import log_softmax, pad
    import math
    import copy
    import time
    from torch.optim.lr_scheduler import LambdaLR
    import pandas as pd
    import altair as alt
    try:
        from torchtext.data.functional import to_map_style_dataset
    except ImportError:
        # torchtext versions without this helper can still use an equivalent
        # local map-style wrapper for iterable datasets.
        from torch.utils.data import Dataset

        class _MapStyleDataset(Dataset):
            def __init__(self, iterator):
                self._data = list(iterator)

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

        def to_map_style_dataset(iterator):
            return _MapStyleDataset(iterator)
    from torch.utils.data import DataLoader
    from torchtext.vocab import build_vocab_from_iterator
    import torchtext.datasets as datasets
    import spacy
    import GPUtil
    import warnings
    from torch.utils.data.distributed import DistributedSampler
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP

    return (torch,)


@app.cell
def _(RUN_EXAMPLES, torch):
    # Some convenience helper functions used throughout the notebook

    def is_interactive_notebook():
        return __name__ == "__main__"


    def show_example(fn, args=[]):
        if __name__ == "__main__" and RUN_EXAMPLES:
            return fn(*args)


    def execute_example(fn, args=[]):
        if __name__ == "__main__" and RUN_EXAMPLES:
            fn(*args)


    class DummyOptimizer(torch.optim.Optimizer):
        def __init__(self):
            self.param_groups = [{"lr": 0}]
            None

        def step(self):
            None

        def zero_grad(self, set_to_none=False):
            None


    class DummyScheduler:
        def step(self):
            None

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
